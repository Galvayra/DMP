from DMP.learning.neuralNet import TensorModel
from DMP.modeling.tfRecorder import TfRecorder
from .variables import *
from DMP.utils.progress_bar import show_progress_bar
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys
from os import path, getcwd
import matplotlib.pyplot as plt

SLIM_PATH = path.dirname(path.abspath(getcwd())) + '/models/research/slim'
sys.path.append(SLIM_PATH)

from preprocessing import vgg_preprocessing

VGG_PATH = 'dataset/images/save/vgg_16.ckpt'


class SlimLearner(TensorModel):
    def __init__(self, tf_record_path):
        super().__init__(is_cross_valid=True)
        self.__tf_record_path = tf_record_path
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()
        self.tf_recorder = TfRecorder(tf_record_path)

    @property
    def tf_record_path(self):
        return self.__tf_record_path

    def __concat_tensor(self, target, prefix):
        for i, img_path in enumerate(sorted(target)):
            tf_img, tf_label = self.tf_recorder.get_img_from_tf_records(img_path)
            tf_img = tf.expand_dims(tf_img, 0)
            tf_label = tf.expand_dims(tf_label, 0)

            if i + 1 == 1:
                self.tf_x = tf_img
                self.tf_y = tf_label
            elif i + 1 == len(target):
                self.tf_x = tf.concat([self.tf_x, tf_img], 0, name=NAME_X + '_' + str(self.num_of_fold))
                self.tf_y = tf.concat([self.tf_y, tf_label], 0, name=NAME_Y + '_' + str(self.num_of_fold))
            else:
                self.tf_x = tf.concat([self.tf_x, tf_img], 0)
                self.tf_y = tf.concat([self.tf_y, tf_label], 0)

            show_progress_bar(i + 1, len(target), prefix="Concatenate tensor for " + prefix)

    def run_fine_tuning(self, x_train, x_test):
        self.num_of_fold += 1
        self.__concat_tensor(x_test, prefix="test ")

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        ################# 신경망 만들기


        print("finish")
        exit(-1)
        tf.reset_default_graph()
        self.clear_tensor()

    def run(self):
        (train_x, train_y), (test_x, test_y) = self.mnist_load()

        #######################
        # 1. placeholder 정의
        x = tf.placeholder(tf.float32, shape=[None, 28, 28])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        is_training = tf.placeholder(tf.bool)

        # ########################
        # # 2. TF-Slim을 이용한 CNN 모델 구현
        # with slim.arg_scope([slim.conv2d],
        #                     padding='SAME',
        #                     activation_fn=tf.nn.elu,
        #                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        #     inputs = tf.reshape(x, [-1, 28, 28, 1])
        #
        #     net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[5, 5], scope='conv1')
        #     net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
        #     net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #     net = slim.flatten(net, scope='flatten3')
        #
        # with slim.arg_scope([slim.fully_connected],
        #                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        #     net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
        #     net = slim.dropout(net, is_training=is_training, scope='dropout3')
        #     outputs = slim.fully_connected(net, 10, activation_fn=None)

        inputs = tf.reshape(x, [-1, 28, 28, 1])
        outputs = self.vgg16(inputs)
        outputs = slim.fully_connected(net, 10, activation_fn=None)

        ########################
        # 3. loss, optimizer, accuracy
        # loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y_))
        # optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # accuracy
        correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ########################
        # 4. Hyper-Paramter 설정 및 데이터 설정
        # Hyper Parameters
        STEPS = 1000
        MINI_BATCH_SIZE = 50

        # tf.data.Dataset을 이용한 배치 크기 만큼 데이터 불러오기
        dataset = tf.data.Dataset.from_tensor_slices(({"image": train_x}, train_y))
        dataset = dataset.shuffle(100000).repeat().batch(MINI_BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()

        ########################
        # Training & Testing
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 학습
            for step in range(STEPS):
                batch_xs, batch_ys = sess.run(next_batch)
                _, cost_val = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs['image'],
                                                                               y_: batch_ys,
                                                                               is_training: True})

            if (step + 1) % 10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs['image'],
                                                               y_: batch_ys,
                                                               is_training: False})
                print("Step : {}, cost : {:.5f}, training accuracy: {:.5f}".format(step + 1, cost_val,
                                                                                   train_accuracy))

            X = test_x.reshape([10, 1000, 28, 28])
            Y = test_y.reshape([10, 1000, 10])

            test_accuracy = np.mean(
                [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], is_training: False}) for i in range(10)])

        print("test accuracy: {:.5f}".format(test_accuracy))

    # @staticmethod
    # def vgg16(inputs):
    #     with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                          activation_fn=tf.nn.relu,
    #                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
    #                          weights_regularizer=slim.l2_regularizer(0.0005)):
    #         net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    #         net = slim.max_pool2d(net, [2, 2], scope='pool1')
    #         net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    #         net = slim.max_pool2d(net, [2, 2], scope='pool2')
    #         net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    #         net = slim.max_pool2d(net, [2, 2], scope='pool3')
    #         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    #         net = slim.max_pool2d(net, [2, 2], scope='pool4')
    #         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    #         net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #         net = slim.fully_connected(net, 4096, scope='fc6')
    #         net = slim.dropout(net, 0.5, scope='dropout6')
    #         net = slim.fully_connected(net, 4096, scope='fc7')
    #         net = slim.dropout(net, 0.5, scope='dropout7')
    #         net = slim.fully_connected(net, 10, activation_fn=None, scope='fc8')
    #         # net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    #         return net
