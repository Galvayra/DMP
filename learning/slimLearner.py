from DMP.learning.neuralNet import TensorModel
from DMP.modeling.tfRecorder import TfRecorder, EXTENSION_OF_TF_RECORD, KEY_OF_TRAIN, KEY_OF_TEST, KEY_OF_DIM, \
    KEY_OF_SHAPE
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
        super().__init__(is_cross_valid=True)
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
            logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1000, is_training=False)

            return logits, end_points

    def __sess_run(self):
        # # 체크포인트로부터 파라미터 복원하기
        # # 마지막 fc8 레이어는 파라미터 복원에서 제외

        # hypothesis = self.__init_cnn_model()
        logits, end_points = self.__init_pre_trained_model()
        fc_7 = end_points['vgg_16/fc7']

        W = tf.Variable(tf.random_normal([4096, 1], mean=0.0, stddev=0.02), name='W')
        b = tf.Variable(tf.random_normal([1], mean=0.0))

        fc_7 = tf.reshape(fc_7, [-1, W.get_shape().as_list()[0]])
        logitx = tf.nn.bias_add(tf.matmul(fc_7, W), b)
        probx = tf.nn.sigmoid(logitx)

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx, labels=self.tf_y))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[W, b])

        init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, slim.get_model_variables('vgg_16'))

        tf_train_record = self.init_tf_record_tensor(key=KEY_OF_TRAIN)
        tf_test_record = self.init_tf_record_tensor(key=KEY_OF_TEST, is_test=True)
        iterator = tf_train_record.make_initializable_iterator()
        next_element = iterator.get_next()
        iterator_test = tf_test_record.make_initializable_iterator()
        next_test_element = iterator_test.get_next()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(iterator.initializer)
            sess.run(iterator_test.initializer)
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "/train", sess.graph)
            saver = tf.train.Saver(max_to_keep=(NUM_OF_LOSS_OVER_FIT + 1))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            init_fn(sess)

            try:
                step = int()
                n_iter = int()
                batch_iter = int(self.tf_recorder.log[KEY_OF_TRAIN + str(self.num_of_fold)] / BATCH_SIZE) + 1

                while not coord.should_stop():
                    n_iter += 1
                    x_batch, y_batch, x_img, x_name = sess.run(next_element)

                    # print(n_iter, x_batch.shape, y_batch.shape, x_img.shape, x_name)
                    _, tra_loss = sess.run(
                        [train_step, cost],
                        feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB},
                        options=run_opts
                    )

                    # 1 epoch
                    if n_iter % batch_iter == 0:
                        step += 1
                        # # if self.do_show and step % NUM_OF_SAVE_EPOCH == 0:
                        print("Step %5d, train loss =  %.5f" % (step, tra_loss))
                        # train_summary, tra_loss, tra_acc = sess.run(
                        #     [merged_summary, cross_entropy, accuracy],
                        #     feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
                        # )

                        # train_writer.add_summary(train_summary, global_step=step)

            except tf.errors.OutOfRangeError:
                # x_test_batch, y_test_batch = self.get_total_batch(sess, next_test_element, is_get_image=True)
                # print(x_test_batch.shape, y_test_batch.shape)
                # prob = sess.run(probx, feed_dict={self.tf_x: x_test_batch, self.tf_y: y_test_batch},
                #                 options=run_opts)
                hypothesis = list()
                y_test = list()
                try:
                    while True:
                        x_batch, y_batch, x_img, tensor_name = sess.run(next_test_element)
                        prob = sess.run(probx, feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: 1})
                        for p, y in zip(prob, y_batch):
                            hypothesis.append(p)
                            y_test.append(y)
                except tf.errors.OutOfRangeError:
                    h = np.array(hypothesis)
                    y_test = np.array(y_test)
                    p = (h > 0.5)
                    # print("\nAccuracy : ", np.sum(p == y_test)*100/float(len(y_test)), '%\n\n\n')
                    saver.save(sess, save_path=self.get_name_of_tensor() + "/model")
            finally:
                coord.request_stop()
                coord.join(threads)

        tf.reset_default_graph()
        self.clear_tensor()

        return h, p, y_test
        # # exculde = ['vgg_16/fc8']
        # # variables_to_restore = slim.get_variables_to_restore(exclude=exculde)
        # # saver = tf.train.Saver(variables_to_restore)
        # # with tf.Session() as sess:
        # #     saver.restore(sess, VGG_PATH)
        # # exit(-1)
        # tf_train_record = self.init_tf_record_tensor(key=KEY_OF_TRAIN)
        # tf_test_record = self.init_tf_record_tensor(key=KEY_OF_TEST, is_test=True)
        # iterator = tf_train_record.make_initializable_iterator()
        # next_element = iterator.get_next()
        # iterator_test = tf_test_record.make_initializable_iterator()
        # next_test_element = iterator_test.get_next()
        #
        # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=self.tf_y))
        # # optimizer
        # train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        # # accuracy
        # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.tf_y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        #
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #     sess.run(iterator.initializer)
        #     sess.run(iterator_test.initializer)
        #
        #     merged_summary = tf.summary.merge_all()
        #     train_writer = tf.summary.FileWriter(self.name_of_log + "/train", sess.graph)
        #     saver = tf.train.Saver(max_to_keep=(NUM_OF_LOSS_OVER_FIT + 1))
        #
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        #     try:
        #         step = int()
        #         n_iter = int()
        #         batch_iter = int(self.tf_recorder.log[KEY_OF_TRAIN + str(self.num_of_fold)] / BATCH_SIZE) + 1
        #
        #         while not coord.should_stop():
        #             n_iter += 1
        #             x_batch, y_batch, x_img, tensor_name = sess.run(next_element)
        #
        #             # # print(x_batch.shape, y_batch.shape, x_img.shape, x_name)
        #             # _, tra_loss = sess.run(
        #             #     [train_step, cross_entropy],
        #             #     feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
        #             # )
        #
        #             # 1 epoch
        #             if n_iter % batch_iter == 0:
        #                 step += 1
        #                 # # if self.do_show and step % NUM_OF_SAVE_EPOCH == 0:
        #                 # print("Step %5d, train loss =  %.5f" % (step, tra_loss))
        #                 # train_summary, tra_loss, tra_acc = sess.run(
        #                 #     [merged_summary, cross_entropy, accuracy],
        #                 #     feed_dict={self.tf_x: x_img, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
        #                 # )
        #                 #
        #                 # train_writer.add_summary(train_summary, global_step=step)
        #
        #     except tf.errors.OutOfRangeError:
        #         x_test_batch, y_test_batch = self.get_test_batch(sess, next_test_element)
        #         print(x_test_batch.shape, y_test_batch.shape)
        #         saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")
        #     finally:
        #         coord.request_stop()
        #         coord.join(threads)
        #
        # tf.reset_default_graph()
        # self.clear_tensor()
        # print("finish")
        # exit(-1)

    def __init_cnn_model(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                             activation_fn=tf.nn.relu,
                             weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(self.tf_x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, self.keep_prob, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, self.keep_prob, scope='dropout7')
            net = slim.fully_connected(net, 1000, scope='fc8')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, self.num_of_output_nodes, activation_fn=tf.nn.sigmoid, scope='output')

            return net

    def clear_tensor(self):
        super().clear_tensor()
        self.tf_name = None

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
