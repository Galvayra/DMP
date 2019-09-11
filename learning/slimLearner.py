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

    # def __concat_tensor(self, target, prefix):
    #     for i, img_path in enumerate(sorted(target)):
    #         tf_img, tf_label = self.tf_recorder.get_img_from_tf_records(img_path)
    #         tf_img = tf.expand_dims(tf_img, 0)
    #         tf_label = tf.expand_dims(tf_label, 0)
    #
    #         if i + 1 == 1:
    #             self.tf_x = tf_img
    #             self.tf_y = tf_label
    #         elif i + 1 == len(target):
    #             self.tf_x = tf.concat([self.tf_x, tf_img], 0, name=NAME_X + '_' + str(self.num_of_fold))
    #             self.tf_y = tf.concat([self.tf_y, tf_label], 0, name=NAME_Y + '_' + str(self.num_of_fold))
    #         else:
    #             self.tf_x = tf.concat([self.tf_x, tf_img], 0)
    #             self.tf_y = tf.concat([self.tf_y, tf_label], 0)
    #
    #         show_progress_bar(i + 1, len(target), prefix="Concatenate tensor for " + prefix)
    #     tf.reset_default_graph()

    def run_fine_tuning(self):
        self.num_of_fold += 1
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=self.shape,
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))

        h, y_predict, y_test = self.__sess_run()
        self.compute_score(y_test, y_predict, h)

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
        """

        :return:
        h = a result of hypothesis (np.array)
        p = a result of predict (np.array)
        y_test = a set of Y test (np.array)

        """
        h, p, y_test = list(), list(), list()
        logits, end_points = self.__init_pre_trained_model()
        fc_7 = end_points['vgg_16/fc7']

        W = tf.Variable(tf.random_normal([4096, 1], mean=0.0, stddev=0.02), name='W')
        b = tf.Variable(tf.random_normal([1], mean=0.0))

        fc_7 = tf.reshape(fc_7, [-1, W.get_shape().as_list()[0]])
        logitx = tf.nn.bias_add(tf.matmul(fc_7, W), b)
        probx = tf.nn.sigmoid(logitx)

        with tf.name_scope(NAME_SCOPE_COST):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx, labels=self.tf_y))
            cost_summ = tf.summary.scalar("cost", cost)

        with tf.name_scope(NAME_SCOPE_PREDICT):
            predict = tf.cast(probx > 0.5, dtype=tf.float32, name=NAME_PREDICT + '_' + str(self.num_of_fold))
            _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
            accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[W, b])
        init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, slim.get_model_variables('vgg_16'))

        # initialize tfRecord
        tf_train_record = self.init_tf_record_tensor(key=KEY_OF_TRAIN)
        tf_valid_record = self.init_tf_record_tensor(key=KEY_OF_VALID, is_test=True)
        tf_test_record = self.init_tf_record_tensor(key=KEY_OF_TEST, is_test=True)

        # initialize iterators
        iterator_train = tf_train_record.make_initializable_iterator()
        iterator_valid = tf_valid_record.make_initializable_iterator()
        iterator_test = tf_test_record.make_initializable_iterator()

        # initialize next element
        next_train_element = iterator_train.get_next()
        next_valid_element = iterator_valid.get_next()
        next_test_element = iterator_test.get_next()

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(iterator_train.initializer)
            sess.run(iterator_valid.initializer)
            sess.run(iterator_test.initializer)

            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "train", sess.graph)
            valid_writer = tf.summary.FileWriter(self.name_of_log + "valid", sess.graph)

            loss_dict = {
                "train": list(),
                "valid": list()
            }
            acc_dict = {
                "train": list(),
                "valid": list()
            }

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            init_fn(sess)
            step = int()

            try:
                n_iter = int()
                batch_iter = int(self.tf_recorder.log[KEY_OF_TRAIN] / BATCH_SIZE) + 1

                while not coord.should_stop():
                    n_iter += 1
                    x_batch, y_batch, x_img, x_name = sess.run(next_train_element)

                    # print(n_iter, x_batch.shape, y_batch.shape, x_img.shape, x_name)
                    train_summary, _, tra_loss, tra_acc = sess.run(
                        [merged_summary, train_step, cost, _accuracy],
                        feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
                    )

                    train_writer.add_summary(train_summary, global_step=n_iter)

                    loss_dict["train"].append(tra_loss)
                    acc_dict["train"].append(tra_acc)

                    # epoch
                    if n_iter % batch_iter == 0:
                        step += 1

                        try:
                            while True:
                                x_valid_batch, y_valid_batch, x_valid_img, x_valid_name = sess.run(next_valid_element)
                                valid_summary, val_loss, val_acc = sess.run(
                                    [merged_summary, cost, _accuracy],
                                    feed_dict={self.tf_x: x_valid_img, self.tf_y: y_valid_batch,
                                               self.keep_prob: KEEP_PROB}
                                )
                                valid_writer.add_summary(valid_summary, global_step=n_iter)
                                loss_dict["valid"].append(val_loss)
                                acc_dict["valid"].append(val_acc)
                        except tf.errors.OutOfRangeError:
                            print(step, len(loss_dict["train"]), len(loss_dict["valid"]))
                        self.__set_average_values(step, loss_dict, acc_dict)

            except tf.errors.OutOfRangeError:
                # last epoch
                if len(loss_dict["train"]) > 0:
                    step += 1
                    self.__set_average_values(step, loss_dict, acc_dict)

                self.save_loss_plot(log_path=self.name_of_log, step_list=[step for step in range(1, step + 1)])

                saver = tf.train.Saver()
                saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                hypothesis = list()
                y_test = list()

                # test scope
                try:
                    while True:
                        x_batch, y_batch, x_img, tensor_name = sess.run(next_test_element)
                        prob = sess.run(probx, feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: 1})
                        for p, y in zip(prob, y_batch):
                            hypothesis.append(p)
                            y_test.append(y)
                except tf.errors.OutOfRangeError:
                    h = np.array(hypothesis)
                    p = (h > 0.5)
                    y_test = np.array(y_test)
            finally:
                coord.request_stop()
                coord.join(threads)

        tf.reset_default_graph()
        self.clear_tensor()

        return h, p, y_test

    def __set_average_values(self, step, loss_dict, acc_dict):
        tra_loss = float(np.mean(np.array(loss_dict["train"])))
        val_loss = float(np.mean(np.array(loss_dict["valid"])))
        tra_acc = float(np.mean(np.array(acc_dict["train"])))
        val_acc = float(np.mean(np.array(acc_dict["valid"])))

        self.tra_loss_list.append(tra_loss)
        self.val_loss_list.append(val_loss)
        self.tra_acc_list.append(tra_acc)
        self.val_acc_list.append(val_acc)

        loss_dict["train"].clear()
        loss_dict["valid"].clear()
        acc_dict["train"].clear()
        acc_dict["valid"].clear()

        print("Step %5d,  train loss = %.5f,  accuracy = %.2f" % (step, tra_loss, tra_acc * 100))
        print("             valid loss = %.5f,  accuracy = %.2f" % (val_loss, val_acc * 100))

    def clear_tensor(self):
        super().clear_tensor()
        self.tf_name = None
