import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import sys
import math
import numpy as np
from .neuralNetModel import EarlyStopping
from DMP.learning.basicLearner import NeuralNet
from os import path, getcwd
from .variables import *

SLIM_PATH = path.dirname(path.abspath(getcwd())) + '/models/research/slim'
sys.path.append(SLIM_PATH)

VGG_PATH = 'dataset/images/ckpt/vgg_16.ckpt'
NUM_OF_EARLY_STOPPING = 5


class TransferLearner(NeuralNet):
    def __init__(self, is_cross_valid=True, model='transfer'):
        super().__init__(is_cross_valid=is_cross_valid)
        self.loss_dict = {
            "train": list(),
            "valid": list()
        }
        self.acc_dict = {
            "train": list(),
            "valid": list()
        }
        self.__model = model
        self.__init_variables()
        self.early_stopping = EarlyStopping(patience=NUM_OF_EARLY_STOPPING, verbose=1)

    @property
    def model(self):
        return self.__model

    def __init_variables(self):
        self.h = list()
        self.p = list()
        self.n_iter = int()

    def __init_place_holder(self, x_train, y_train):
        self.num_of_fold += 1
        self.__init_variables()
        self.num_of_input_nodes = len(x_train[0])
        self.num_of_output_nodes = len(y_train[0])
        
        shape = [None] + [dim for dim in x_train[0].shape]
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=shape,
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))

    def transfer_learning(self, x_train, y_train, x_test, y_test):
        # self.show_sets(y_train, y_test)
        self.__init_place_holder(x_train, y_train)

        self.__sess_run_for_transfer(x_train, y_train, x_test, y_test)
        self.compute_score(x_test, self.p, self.h)

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
        # vgg_scope
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

        if self.model == "transfer":
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[fc])
        else:
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        with tf.Session() as sess:
            self.merged_summary = tf.summary.merge_all()
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

            init_fn(sess)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            self.batch_iter = int(math.ceil(len(x_train) / BATCH_SIZE))

            for step in range(1, self.best_epoch + 1):
                # training scope
                for i in range(self.batch_iter):
                    self.n_iter += 1
                    batch_x = x_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                    batch_y = y_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

                    _, tra_loss = sess.run(
                        [train_step, cost],
                        feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: KEEP_PROB}
                    )

                if not self.is_cross_valid:
                    self.__run_cost_func(sess, x_test, y_test, val_writer, cost, _accuracy, key="valid")
                self.__run_cost_func(sess, x_train, y_train, train_writer, cost, _accuracy)

                if self.__set_average_values(step):
                    early_stop = True
                    saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")
                else:
                    early_stop = False

                if self.do_show and step % NUM_OF_SHOW_EPOCH == 0:
                    saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                if early_stop:
                    break

            self.__set_test_prob(sess, x_test, y_test, hypothesis)

        tf.reset_default_graph()

    def __run_cost_func(self, sess, x_train, y_train, summary_writer, cost_tensor, acc_tensor, key="train"):
        for i in range(self.batch_iter):
            batch_x = x_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
            batch_y = y_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

            summary, tra_loss, tra_acc = sess.run(
                [self.merged_summary, cost_tensor, acc_tensor],
                feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: KEEP_PROB}
            )

            summary_writer.add_summary(summary, global_step=self.n_iter)
            self.loss_dict[key].append(tra_loss)
            self.acc_dict[key].append(tra_acc)

    def __set_average_values(self, step):
        tra_loss = float(np.mean(np.array(self.loss_dict["train"])))
        tra_acc = float(np.mean(np.array(self.acc_dict["train"])))
        self.tra_loss_list.append(tra_loss)
        self.tra_acc_list.append(tra_acc)
        self.loss_dict["train"].clear()
        self.acc_dict["train"].clear()
        print("Step %5d,  train loss = %.5f,  accuracy = %.2f" % (step, tra_loss, tra_acc * 100))

        if not self.is_cross_valid:
            val_loss = float(np.mean(np.array(self.loss_dict["valid"])))
            val_acc = float(np.mean(np.array(self.acc_dict["valid"])))
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
            self.loss_dict["valid"].clear()
            self.acc_dict["valid"].clear()
            print("             valid loss = %.5f,  accuracy = %.2f" % (val_loss, val_acc * 100))

            if USE_EARLY_STOPPING and self.early_stopping.validate(val_loss, val_acc):
                return True

        return False

    def __set_test_prob(self, sess, x_test, y_test, hypothesis):
        h_list = list()

        for i in range(self.batch_iter):
            batch_x = x_test[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
            batch_y = y_test[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

            for h in sess.run(hypothesis, feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: 1}):
                h_list.append(h)

        self.h = np.array(h_list)
        self.p = (self.h > 0.5)
