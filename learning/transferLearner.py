import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import sys
import math
from DMP.learning.basicLearner import NeuralNet
from os import path, getcwd
from .variables import *

SLIM_PATH = path.dirname(path.abspath(getcwd())) + '/models/research/slim'
sys.path.append(SLIM_PATH)

VGG_PATH = 'dataset/images/ckpt/vgg_16.ckpt'


class TransferLearner(NeuralNet):
    def __init__(self, is_cross_valid=True):
        super().__init__(is_cross_valid=is_cross_valid)

    def __init_var_result(self):
        self.h = list()
        self.p = list()
        self.y_test = list()

    def __init_place_holder(self, x_train, y_train):
        self.num_of_fold += 1
        self.num_of_input_nodes = len(x_train[0])
        self.num_of_output_nodes = len(y_train[0])
        
        shape = [None] + [dim for dim in x_train[0].shape]
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=shape,
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))

    def transfer_learning(self, x_train, y_train, x_test, y_test):
        self.__init_place_holder(x_train, y_train)

        h, y_predict, accuracy = self.__sess_run_for_transfer(x_train, y_train, x_test, y_test)
        self.compute_score(x_test, y_predict, h, accuracy)

        if self.is_cross_valid:
            key = KEY_TEST
        else:
            key = KEY_VALID

        self.set_score(target=key)
        self.show_score(target=key)

    def __init_pre_trained_model(self):
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1, is_training=False)
            # logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1000, is_training=is_training)

            return logits, end_points

    def __sess_run_for_transfer(self, x_train, y_train, x_test, y_test):
        logits, end_points = self.__init_pre_trained_model()
        exclude = ['vgg_16/fc8']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, variables_to_restore)

        str_n_fold = '_' + str(self.num_of_fold)
        fc = tf.contrib.framework.get_variables('vgg_16/fc8')

        with tf.name_scope(NAME_SCOPE_COST):
            hypothesis = tf.nn.sigmoid(logits, name=NAME_HYPO + str_n_fold)
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.tf_y))
            cost_summary = tf.summary.scalar("cost", cost)

        with tf.name_scope(NAME_SCOPE_PREDICT):
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT + str_n_fold)
            _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", _accuracy)

        # if self.model == "transfer":
        #     train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[fc])
        # else:
        #     train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[fc])

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            print("\n\n\n")

            if not self.is_cross_valid:
                train_writer = tf.summary.FileWriter(self.name_of_log + "/train",
                                                     sess.graph)
                val_writer = tf.summary.FileWriter(self.name_of_log + "/val",
                                                   sess.graph)
                saver = tf.train.Saver(max_to_keep=(NUM_OF_LOSS_OVER_FIT + 1))
            else:
                train_writer = tf.summary.FileWriter(self.name_of_log + "/fold_" + str(self.num_of_fold) + "/train",
                                                     sess.graph)
                val_writer = None
                saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            batch_iter = int(math.ceil(len(x_train) / BATCH_SIZE))

            for step in range(1, self.best_epoch + 1):
                # mini-batch
                for i in range(batch_iter):
                    batch_x = x_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                    batch_y = y_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

                    _, tra_loss = sess.run(
                        [train_step, cost],
                        feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: KEEP_PROB}
                    )

                # training
                if self.do_show and step % NUM_OF_SAVE_EPOCH == 0:
                    if self.is_cross_valid:
                        train_summary, tra_loss, tra_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                        )

                        train_writer.add_summary(train_summary, global_step=step)
                        print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc * 100.0))

                        saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")
                    else:
                        train_summary, tra_loss, tra_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                        )

                        train_writer.add_summary(train_summary, global_step=step)
                        print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc * 100.0))

                        val_summary, val_loss, val_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_test, self.tf_y: y_test, self.keep_prob: KEEP_PROB}
                        )

                        # write validation curve on tensor board
                        val_writer.add_summary(val_summary, global_step=step)
                        print("            valid loss =  %.5f, valid  acc = %.2f" % (val_loss, val_acc * 100.0))

                        # save tensor every NUM_OF_SAVE_EPOCH
                        saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                        if self.__is_stopped_training(val_loss):
                            break

            h, p, acc = sess.run([hypothesis, predict, _accuracy],
                                 feed_dict={self.tf_x: x_test, self.tf_y: y_test, self.keep_prob: 1.0})

        tf.reset_default_graph()

        return h, p, acc
