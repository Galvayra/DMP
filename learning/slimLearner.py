from DMP.learning.neuralNet import TensorModel
from DMP.modeling.tfRecorder import TfRecorder, KEY_OF_TRAIN, KEY_OF_TEST, KEY_OF_VALID, KEY_OF_SHAPE
from .variables import *
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.slim.nets
import sys
from os import path, getcwd

SLIM_PATH = path.dirname(path.abspath(getcwd())) + '/models/research/slim'
sys.path.append(SLIM_PATH)

VGG_PATH = 'dataset/images/ckpt/vgg_16.ckpt'


class SlimLearner(TensorModel):
    def __init__(self):
        super().__init__(is_cross_valid=False)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = 1
        self.tf_recorder = TfRecorder(self.tf_record_path)
        self.tf_name = None
        self.tf_recorder.do_encode_image = True

        # self.shape = (None, Width, Height, channels)
        shape = self.tf_recorder.log[KEY_OF_SHAPE][:]
        shape.insert(0, None)
        self.shape = shape

        self.loss_dict = {
            "train": list(),
            "valid": list()
        }
        self.acc_dict = {
            "train": list(),
            "valid": list()
        }

    def __init_var_result(self):
        self.h = list()
        self.p = list()
        self.y_test = list()

    def run_fine_tuning(self):
        self.num_of_fold += 1
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=self.shape,
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))

        self.__init_var_result()
        self.__sess_run()
        self.compute_score(self.y_test, self.p, self.h)

        if self.is_cross_valid:
            key = KEY_TEST
        else:
            key = KEY_VALID

        self.set_score(target=key)
        self.show_score(target=key)

    def __init_pre_trained_model(self):
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1000, is_training=True)

            return logits, end_points

    def __sess_run(self):
        logits, end_points = self.__init_pre_trained_model()
        fc_7 = end_points['vgg_16/fc7']

        W = tf.Variable(tf.random_normal([4096, 1], mean=0.0, stddev=0.02), name='W')
        b = tf.Variable(tf.random_normal([1], mean=0.0))

        fc_7 = tf.reshape(fc_7, [-1, W.get_shape().as_list()[0]])
        logitx = tf.nn.bias_add(tf.matmul(fc_7, W), b)
        hypothesis = tf.nn.sigmoid(logitx)

        with tf.name_scope(NAME_SCOPE_COST):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx, labels=self.tf_y))
            cost_summary = tf.summary.scalar("cost", cost)

        with tf.name_scope(NAME_SCOPE_PREDICT):
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT + '_' + str(self.num_of_fold))
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", acc)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[W, b])
        init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, slim.get_model_variables('vgg_16'))

        # initialize tfRecord and iterators
        tf_train_record = self.init_tf_record_tensor(key=KEY_OF_TRAIN)
        tf_test_record = self.init_tf_record_tensor(key=KEY_OF_TEST, is_test=True)
        iterator_train = tf_train_record.make_initializable_iterator()
        iterator_test = tf_test_record.make_initializable_iterator()

        if not self.is_cross_valid:
            tf_valid_record = self.init_tf_record_tensor(key=KEY_OF_VALID, is_test=True)
            iterator_valid = tf_valid_record.make_initializable_iterator()
        else:
            iterator_valid = None

        # initialize next element
        next_train_element = iterator_train.get_next()

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(iterator_train.initializer)

            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "train", sess.graph)
            valid_writer = tf.summary.FileWriter(self.name_of_log + "valid", sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            init_fn(sess)
            step = int()

            try:
                n_iter = int()
                batch_iter = int(self.tf_recorder.log[KEY_OF_TRAIN] / BATCH_SIZE) + 1

                # Training scope
                while not coord.should_stop():
                    n_iter += 1
                    x_batch, y_batch, x_img, x_name = sess.run(next_train_element)

                    train_summary, _, tra_loss, tra_acc = sess.run(
                        [merged_summary, train_step, cost, acc],
                        feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
                    )

                    train_writer.add_summary(train_summary, global_step=n_iter)

                    self.loss_dict["train"].append(tra_loss)
                    self.acc_dict["train"].append(tra_acc)

                    # epoch
                    if n_iter % batch_iter == 0:
                        step += 1
                        self.__set_valid_loss(sess, n_iter, iterator_valid, merged_summary, cost, acc, valid_writer)
                        self.__set_average_values(step)

            except tf.errors.OutOfRangeError:
                # last epoch
                if len(self.loss_dict["train"]) > 0:
                    step += 1
                    self.__set_valid_loss(sess, n_iter, iterator_valid, merged_summary, cost, acc, valid_writer)
                    self.__set_average_values(step)

                self.save_loss_plot(log_path=self.name_of_log, step_list=[step for step in range(1, step + 1)])

                saver = tf.train.Saver()
                saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                # set self.h, self.p, self.y_test
                self.__set_test_prob(sess, iterator_test, hypothesis)
            finally:
                coord.request_stop()
                coord.join(threads)

        tf.reset_default_graph()
        self.clear_tensor()

    # Validation scope
    def __set_valid_loss(self, sess, n_iter, iterator, merged_summary, cost, accuracy, valid_writer):
        if not self.is_cross_valid:
            sess.run(iterator.initializer)
            next_element = iterator.get_next()
            try:
                while True:
                    x_valid_batch, y_valid_batch, x_valid_img, x_valid_name = sess.run(next_element)
                    valid_summary, val_loss, val_acc = sess.run(
                        [merged_summary, cost, accuracy],
                        feed_dict={self.tf_x: x_valid_img, self.tf_y: y_valid_batch,
                                   self.keep_prob: KEEP_PROB}
                    )
                    valid_writer.add_summary(valid_summary, global_step=n_iter)
                    self.loss_dict["valid"].append(val_loss)
                    self.acc_dict["valid"].append(val_acc)
            except tf.errors.OutOfRangeError:
                print(len(self.loss_dict["train"]), len(self.loss_dict["valid"]))

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

    def __set_test_prob(self, sess, iterator, hypothesis):
        h_list = list()
        y_test = list()

        # Test scope
        sess.run(iterator.initializer)
        next_element = iterator.get_next()
        try:
            while True:
                x_batch, y_batch, x_img, tensor_name = sess.run(next_element)
                h_batch = sess.run(hypothesis, feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: 1})
                for h, y in zip(h_batch, y_batch):
                    h_list.append(h)
                    y_test.append(y)
        except tf.errors.OutOfRangeError:
            self.h = np.array(h_list)
            self.p = (self.h > 0.5)
            self.y_test = np.array(y_test)

    def clear_tensor(self):
        super().clear_tensor()
        self.tf_name = None
