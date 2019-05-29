import tensorflow as tf
from .variables import *
from .score import MyScore
import os
import shutil
import math
import sys
import json

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, LOG_DIR_NAME, LEARNING_RATE
else:
    from DMP.utils.arg_predict import DO_SHOW, DO_DELETE, LOG_DIR_NAME

BATCH_SIZE = 512


class MyNeuralNetwork(MyScore):
    def __init__(self, is_cross_valid=True):
        super().__init__()
        self.tf_x = None
        self.tf_y = None
        self.keep_prob = None
        self.hypothesis = None
        self.best_epoch = int()
        self.num_of_dimension = int()
        self.num_of_hidden = float()
        self.learning_rate = float()
        self.__loss_list = list()
        self.__name_of_log = str()
        self.__name_of_tensor = str()
        self.__is_cross_valid = is_cross_valid
        self.__init_log_and_tensor()

    @property
    def loss_list(self):
        return self.__loss_list

    @property
    def name_of_log(self):
        return self.__name_of_log

    @name_of_log.setter
    def name_of_log(self, name):
        self.__name_of_log = name

    @property
    def name_of_tensor(self):
        return self.__name_of_tensor

    @name_of_tensor.setter
    def name_of_tensor(self, name):
        self.__name_of_tensor = name

    @property
    def is_cross_valid(self):
        return self.__is_cross_valid

    def __init_log_and_tensor(self):
        self.name_of_log = PATH_LOGS + LOG_DIR_NAME
        self.name_of_tensor = PATH_TENSOR + LOG_DIR_NAME

        if DO_DELETE:
            if os.path.isdir(self.name_of_log):
                shutil.rmtree(self.name_of_log)
            os.mkdir(self.name_of_log)

            if os.path.isdir(self.name_of_tensor):
                shutil.rmtree(self.name_of_tensor)
            os.mkdir(self.name_of_tensor)

    def __set_name_of_log(self):
        name_of_log = self.name_of_log + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_log)

        if DO_SHOW:
            print("======== Directory for Saving ========")
            print("   Log File -", name_of_log)

    def __set_name_of_tensor(self):
        name_of_tensor = self.name_of_tensor + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_tensor)

        if DO_SHOW:
            print("Tensor File -", name_of_tensor, "\n\n\n")

    def __get_name_of_tensor(self):
        return self.name_of_tensor + "fold_" + str(self.num_of_fold)

    def __init_feed_forward_layer(self, num_input_node, input_layer):
        if NUM_HIDDEN_DIMENSION:
            num_hidden_node = NUM_HIDDEN_DIMENSION
        else:
            num_hidden_node = num_input_node

        tf_weight = list()
        tf_bias = list()
        tf_layer = [input_layer]

        # # make hidden layers
        for i in range(NUM_HIDDEN_LAYER):
            # set number of hidden node
            num_hidden_node = int(num_input_node / RATIO_HIDDEN)

            # append weight
            tf_weight.append(tf.get_variable(name="h_weight_" + str(i + 1) + '_' + str(self.num_of_fold),
                                             dtype=tf.float32,
                                             shape=[num_input_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            # append bias
            tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]),
                                       name="h_bias_" + str(i + 1) + '_' + str(self.num_of_fold)))
            layer = tf.add(tf.matmul(tf_layer[i], tf_weight[i]), tf_bias[i])

            # append hidden layer
            hidden_layer = tf.nn.relu(layer)
            tf_layer.append(tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob,
                                          name="dropout_" + str(i + 1) + '_' + str(self.num_of_fold)))

            # set number of node which is next layer
            num_input_node = int(num_input_node / RATIO_HIDDEN)

        tf_weight.append(tf.get_variable(dtype=tf.float32, shape=[num_hidden_node, 1],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         name="o_weight_" + str(self.num_of_fold)))
        tf_bias.append(tf.Variable(tf.random_normal([1]),
                                   name="o_bias_" + str(self.num_of_fold)))

        if DO_SHOW:
            print("\n\n======== Feed Forward Layer ========")
            for i, layer in enumerate(tf_layer):
                print("Layer", i + 1, "-", layer.shape)

        # return X*W + b
        return tf.add(tf.matmul(tf_layer[-1], tf_weight[-1]), tf_bias[-1])

    def feed_forward(self, x_train, y_train, x_valid, y_valid):
        num_of_dimension = len(x_train[0])

        self.num_of_fold += 1
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension],
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name=NAME_PROB + '_' + str(self.num_of_fold))

        # initialize neural network
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=self.tf_x)
        h, y_predict, accuracy = self.__sess_run(hypothesis, x_train, y_train, x_valid, y_valid)
        self.compute_score(y_valid, y_predict, h, accuracy)

        if self.is_cross_valid:
            key = KEY_TEST
        else:
            key = KEY_VALID

        self.set_score(target=key)
        self.__show_score(target=key)

    def __init_convolution_layer(self, num_of_dimension):
        num_of_image = int(math.sqrt(num_of_dimension))
        num_of_filter = [16, 32]
        size_of_filter = 3

        tf_x_img = tf.reshape(self.tf_x, [-1, num_of_image, num_of_image, 1])

        filter_1 = tf.Variable(
            tf.random_normal([size_of_filter, size_of_filter, 1, num_of_filter[0]], stddev=0.01),
            name="cnn_filter_1")
        layer_1 = tf.nn.conv2d(tf_x_img, filter_1, strides=[1, 1, 1, 1], padding="SAME", name="cnn_layer_1")
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                 name="cnn_pooling_1")
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob, name="cnn_dropout_1")
        num_of_image = math.ceil(num_of_image / 2)

        filter_2 = tf.Variable(
            tf.random_normal([size_of_filter, size_of_filter, num_of_filter[0], num_of_filter[1]], stddev=0.01),
            name="cnn_filter_2")
        layer_2 = tf.nn.conv2d(layer_1, filter_2, strides=[1, 1, 1, 1], padding="SAME", name="cnn_layer_2")
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                 name="cnn_pooling_2")
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob, name="cnn_dropout_2")
        num_of_image = math.ceil(num_of_image / 2)

        num_of_dimension = num_of_image * num_of_image * num_of_filter[-1]
        convolution_layer = tf.reshape(layer_2, [-1, num_of_dimension], name="cnn_span_layer")

        if DO_SHOW:
            print("\n\n======== Convolution Layer ========")
            print("tf_x     -", self.tf_x.shape)
            print("tf_x_img -", tf_x_img.shape)

            print("\n\nfilter_1 -", filter_1.shape)
            print(" layer_1 -", layer_1.shape)
            print("\n\nfilter_2 -", filter_2.shape)
            print(" layer_2 -", layer_2.shape)

        return convolution_layer, num_of_dimension

    # The model of Paper 'Deep Learning for the Classification of Lung Nodules'
    def __init_convolution_layer_model_1(self, num_of_dimension):
        num_of_image = int(math.sqrt(num_of_dimension))
        num_of_filter = [20, 50, 500]
        size_of_filter = 7

        tf_x_img = tf.reshape(self.tf_x, [-1, num_of_image, num_of_image, 1])

        # 7 x 7 x 1 x 20
        filter_1 = tf.Variable(
            tf.random_normal([size_of_filter, size_of_filter, 1, num_of_filter[0]], stddev=0.01),
            name="cnn_filter_1")
        conv_1 = tf.nn.conv2d(tf_x_img, filter_1, strides=[1, 1, 1, 1], padding="VALID",
                              name="conv_1_" + str(self.num_of_fold))
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID",
                                name="pool_1_" + str(self.num_of_fold))
        pool_1 = tf.nn.dropout(pool_1, keep_prob=self.keep_prob,
                               name="dropout_1_" + str(self.num_of_fold))

        # 7 x 7 x 20 x 50
        filter_2 = tf.Variable(
            tf.random_normal([size_of_filter, size_of_filter, num_of_filter[0], num_of_filter[1]], stddev=0.01),
            name="cnn_filter_2")
        conv_2 = tf.nn.conv2d(pool_1, filter_2, strides=[1, 1, 1, 1], padding="VALID",
                              name="conv_2_" + str(self.num_of_fold))
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID",
                                name="pool_2_" + str(self.num_of_fold))
        pool_2 = tf.nn.dropout(pool_2, keep_prob=self.keep_prob,
                               name="dropout_2_" + str(self.num_of_fold))

        # 7 x 7 x 50 x 500
        filter_3 = tf.Variable(
            tf.random_normal([size_of_filter, size_of_filter, num_of_filter[1], num_of_filter[2]], stddev=0.01),
            name="cnn_filter_32")
        conv_3 = tf.nn.conv2d(pool_2, filter_3, strides=[1, 1, 1, 1], padding="VALID",
                              name="conv_3_" + str(self.num_of_fold))
        pool_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID",
                                name="pool_3_" + str(self.num_of_fold))
        pool_3 = tf.nn.dropout(pool_3, keep_prob=self.keep_prob,
                               name="dropout_3_" + str(self.num_of_fold))

        relu_layer = tf.nn.relu(pool_3)

        convolution_layer = tf.reshape(relu_layer, [-1, num_of_filter[-1]],
                                       name="cnn_span_layer_" + str(self.num_of_fold))

        if DO_SHOW:
            print("\n\n======== Convolution Layer ========")
            print("tf_x     -", self.tf_x.shape)
            print("tf_x_img -", tf_x_img.shape)

            print("\n\nconv_1 -", conv_1.shape)
            print("pool_1 -", pool_1.shape)
            print("\n\nconv_2 -", conv_2.shape)
            print("pool_2 -", pool_2.shape)
            print("\n\nconv_3 -", conv_3.shape)
            print("pool_3 -", pool_3.shape)
            print("\n\ncnn_span_layer -", convolution_layer.shape)

        return convolution_layer, num_of_filter[-1]

    def convolution(self, x_train, y_train, x_valid, y_valid):
        num_of_dimension = len(x_train[0])

        self.num_of_fold += 1
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension],
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name=NAME_PROB + '_' + str(self.num_of_fold))

        # concat CNN to Feed Forward NN
        convolution_layer, num_of_dimension = self.__init_convolution_layer_model_1(num_of_dimension)
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=convolution_layer)
        h, y_predict, accuracy = self.__sess_run(hypothesis, x_train, y_train, x_valid, y_valid)
        self.compute_score(y_valid, y_predict, h, accuracy)

        if self.is_cross_valid:
            key = KEY_TEST
        else:
            key = KEY_VALID

        self.set_score(target=key)
        self.__show_score(target=key)

    def __sess_run(self, hypothesis, x_train, y_train, x_valid, y_valid):
        if DO_SHOW:
            print("Layer O -", hypothesis.shape, "\n\n\n")
        hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO + '_' + str(self.num_of_fold))

        with tf.name_scope("cost"):
            cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
            cost_summ = tf.summary.scalar("cost", cost)

        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        # cut off
        predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT + '_' + str(self.num_of_fold))
        _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        # set file names for saving
        self.__set_name_of_log()
        self.__set_name_of_tensor()
        tf.Variable(LEARNING_RATE, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(NUM_HIDDEN_LAYER, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

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

            for step in range(1, EPOCH + 1):
                # mini-batch
                for i in range(batch_iter):
                    batch_x = x_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                    batch_y = y_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

                    _, tra_loss = sess.run(
                        [train_op, cost],
                        feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: KEEP_PROB}
                    )

                # training
                if DO_SHOW and step % NUM_OF_SAVE_EPOCH == 0:
                    if not self.is_cross_valid:
                        train_summary, tra_loss, tra_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                        )

                        train_writer.add_summary(train_summary, global_step=step)
                        print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc * 100.0))

                        val_summary, val_loss, val_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_valid, self.tf_y: y_valid, self.keep_prob: KEEP_PROB}
                        )

                        # write validation curve on tensor board
                        val_writer.add_summary(val_summary, global_step=step)
                        print("            valid loss =  %.5f, valid  acc = %.2f" % (val_loss, val_acc*100.0))

                        # save tensor every NUM_OF_SAVE_EPOCH
                        saver.save(sess, global_step=step, save_path=self.__get_name_of_tensor() + "/model")

                        if self.__is_stopped_training(val_loss):
                            break
                    else:
                        train_summary, tra_loss, tra_acc = sess.run(
                            [merged_summary, cost, _accuracy],
                            feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                        )

                        train_writer.add_summary(train_summary, global_step=step)
                        print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc * 100.0))

                        saver.save(sess, global_step=step, save_path=self.__get_name_of_tensor() + "/model")

            h, p, acc = sess.run([hypothesis, predict, _accuracy],
                                 feed_dict={self.tf_x: x_valid, self.tf_y: y_valid, self.keep_prob: 1.0})

        tf.reset_default_graph()

        return h, p, acc

    def __is_stopped_training(self, val_loss):
        self.loss_list.append(val_loss)

        cnt_train = len(self.loss_list)

        if cnt_train < 5:
            return False
        else:
            loss_default = self.loss_list[cnt_train - (NUM_OF_LOSS_OVER_FIT + 1)]
            cnt_loss_over_fit = int()

            for loss in self.loss_list[cnt_train - NUM_OF_LOSS_OVER_FIT:cnt_train]:
                if loss > loss_default:
                    cnt_loss_over_fit += 1
                # loss_default = loss

            if cnt_loss_over_fit == NUM_OF_LOSS_OVER_FIT:
                return True
            else:
                return False

    def load_nn(self, x_test, y_test):
        self.num_of_fold += 1
        checkpoint = tf.train.get_checkpoint_state(self.__get_name_of_tensor())
        paths = checkpoint.all_model_checkpoint_paths
        path = paths[len(paths) - (NUM_OF_LOSS_OVER_FIT + 1)]
        self.best_epoch = int(path.split("/")[-1].split("model-")[-1])
        self.num_of_dimension = len(x_test[0])

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(sess, path)

            print("\n\n\ncheckpoint -", path, "\nBest Epoch  -", self.best_epoch, "\n")

            # load tensor
            graph = tf.get_default_graph()
            tf_x = graph.get_tensor_by_name(NAME_X + "_" + str(self.num_of_fold) + ":0")
            tf_y = graph.get_tensor_by_name(NAME_Y + "_" + str(self.num_of_fold) + ":0")
            keep_prob = graph.get_tensor_by_name(NAME_PROB + "_" + str(self.num_of_fold) + ":0")
            hypothesis = graph.get_tensor_by_name(NAME_HYPO + "_" + str(self.num_of_fold) + ":0")
            predict = graph.get_tensor_by_name(NAME_PREDICT + "_" + str(self.num_of_fold) + ":0")
            num_of_hidden = graph.get_tensor_by_name(NAME_HIDDEN + "_" + str(self.num_of_fold) + ":0")
            learning_rate = graph.get_tensor_by_name(NAME_LEARNING_RATE + "_" + str(self.num_of_fold) + ":0")

            self.num_of_hidden, self.learning_rate = sess.run([num_of_hidden, learning_rate])
            h, y_predict = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test, keep_prob: 1})

        return h, y_predict

    def predict(self, h, y_predict, y_test):
        def __get_reverse(_y_labels, is_hypothesis=False):
            _y_labels_reverse = list()

            if is_hypothesis:
                for _y in _y_labels:
                    _y_labels_reverse.append([1 - _y[0]])
            else:
                for _y in _y_labels:
                    if _y == [0]:
                        _y_labels_reverse.append([1])
                    else:
                        _y_labels_reverse.append([0])

            return _y_labels_reverse

        # set score of immortality
        self.compute_score(__get_reverse(y_test), __get_reverse(y_predict), __get_reverse(h, is_hypothesis=True))
        self.set_score(target=KEY_IMMORTALITY)

        # set score of mortality
        self.compute_score(y_test, y_predict, h)
        self.set_score(target=KEY_MORTALITY)

        # set 2 class score
        self.set_2_class_score()

        if self.is_cross_valid:
            self.show_performance()

        self.set_plot()

    def set_multi_plot(self):

        title = "FFNN_baseline"

        fpr = [0., 0., 0.26246719, 0.26246719, 0.52493438,
               0.52493438, 1.04986877, 1.04986877, 1.31233596, 1.31233596,
               3.1496063, 3.1496063, 3.67454068, 3.67454068, 3.93700787,
               3.93700787, 4.72440945, 4.72440945, 5.24934383, 5.24934383,
               5.77427822, 5.77427822, 6.56167979, 6.56167979, 7.34908136,
               7.34908136, 7.61154856, 7.61154856, 17.32283465, 17.32283465,
               18.37270341, 18.37270341, 21.52230971, 21.52230971, 22.04724409,
               22.04724409, 22.83464567, 22.83464567, 24.67191601, 24.67191601,
               36.22047244, 36.22047244, 49.08136483, 49.08136483, 50.1312336,
               50.1312336, 65.09186352, 65.09186352, 100.]

        tpr = [2.7027027, 5.40540541, 5.40540541, 16.21621622, 16.21621622,
               18.91891892, 18.91891892, 24.32432432, 24.32432432, 32.43243243,
               32.43243243, 35.13513514, 35.13513514, 48.64864865, 48.64864865,
               51.35135135, 51.35135135, 54.05405405, 54.05405405, 56.75675676,
               56.75675676, 62.16216216, 62.16216216, 64.86486486, 64.86486486,
               70.27027027, 70.27027027, 72.97297297, 72.97297297, 75.67567568,
               75.67567568, 78.37837838, 78.37837838, 81.08108108, 81.08108108,
               83.78378378, 83.78378378, 86.48648649, 86.48648649, 89.18918919,
               89.18918919, 91.89189189, 91.89189189, 94.59459459, 94.59459459,
               97.2972973, 97.2972973, 100., 100.]

        self.set_plot(fpr=fpr, tpr=tpr, title=title)

    def save(self, data_handler=False):
        # set total score of immortality and mortality

        self.set_performance()
        self.show_performance()

        if self.is_cross_valid:
            pass
        else:
            self.save_score(data_handler=data_handler,
                            best_epoch=self.best_epoch,
                            num_of_dimension=self.num_of_dimension,
                            num_of_hidden=self.num_of_hidden,
                            learning_rate=self.learning_rate)

    def save_process_time(self):
        with open(self.name_of_log + FILE_OF_TRAINING_TIME, 'w') as outfile:
            json.dump(self.show_process_time(), outfile, indent=4)
