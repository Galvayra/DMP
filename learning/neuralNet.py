import tensorflow as tf
from .variables import *
from .score import MyScore
import os
import shutil
import math
import sys
import numpy as np

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, LOG_DIR_NAME, LEARNING_RATE
else:
    from DMP.utils.arg_predict import DO_SHOW, DO_DELETE, LOG_DIR_NAME


class MyNeuralNetwork(MyScore):
    def __init__(self):
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

    def __set_name_of_log(self):
        log_name = PATH_LOGS + LOG_DIR_NAME

        if DO_DELETE:
            if os.path.isdir(log_name):
                shutil.rmtree(log_name)
            os.mkdir(log_name)

        if DO_SHOW:
            print("======== Directory for Saving ========")
            print("   Log File -", log_name)

        self.name_of_log = log_name

    def __set_name_of_tensor(self):
        tensor_name = PATH_TENSOR + LOG_DIR_NAME

        if DO_DELETE:
            if os.path.isdir(tensor_name):
                shutil.rmtree(tensor_name)
            os.mkdir(tensor_name)

        if DO_SHOW:
            print("Tensor File -", tensor_name, "\n\n\n")

        self.name_of_tensor = tensor_name

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
            tf_weight.append(tf.get_variable(name="h_weight_" + str(i + 1), dtype=tf.float32,
                                             shape=[num_input_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            # append bias
            tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i + 1)))
            layer = tf.add(tf.matmul(tf_layer[i], tf_weight[i]), tf_bias[i])

            # append hidden layer
            hidden_layer = tf.nn.relu(layer)
            tf_layer.append(tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob, name="dropout_" + str(i + 1)))

            # set number of node which is next layer
            num_input_node = int(num_input_node / RATIO_HIDDEN)

        tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                         initializer=tf.contrib.layers.xavier_initializer()))
        tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

        if DO_SHOW:
            print("\n\n======== Feed Forward Layer ========")
            for i, layer in enumerate(tf_layer):
                print("Layer", i + 1, "-", layer.shape)

        # return X*W + b
        return tf.add(tf.matmul(tf_layer[-1], tf_weight[-1]), tf_bias[-1])

    def feed_forward(self, x_train, y_train, x_valid, y_valid):
        num_of_dimension = len(x_train[0])

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension], name=NAME_X)
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB)

        # initialize neural network
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=self.tf_x)
        h, y_predict, accuracy = self.__sess_run(hypothesis, x_train, y_train, x_valid, y_valid)
        self.compute_score(y_valid, y_predict, h, accuracy)
        self.set_score(target=KEY_VALID)
        self.show_score(target=KEY_VALID)

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

    def convolution(self, x_train, y_train, x_valid, y_valid):
        num_of_dimension = len(x_train[0])

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension], name=NAME_X)
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB)

        # concat CNN to Feed Forward NN
        convolution_layer, num_of_dimension = self.__init_convolution_layer(num_of_dimension)
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=convolution_layer)
        h, y_predict, accuracy = self.__sess_run(hypothesis, x_train, y_train, x_valid, y_valid)
        self.compute_score(y_valid, y_predict, h, accuracy)
        self.set_score(target=KEY_VALID)
        self.show_score(target=KEY_VALID)

    def __sess_run(self, hypothesis, x_train, y_train, x_valid, y_valid):
        if DO_SHOW:
            print("Layer O -", hypothesis.shape, "\n\n\n")
        hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO)

        with tf.name_scope("cost"):
            cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
            cost_summ = tf.summary.scalar("cost", cost)

        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        # cut off
        predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
        _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        # set file names for saving
        self.__set_name_of_log()
        self.__set_name_of_tensor()
        tf.Variable(LEARNING_RATE, name=NAME_LEARNING_RATE)
        tf.Variable(NUM_HIDDEN_LAYER, name=NAME_HIDDEN)

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "/train", sess.graph)
            val_writer = tf.summary.FileWriter(self.name_of_log + "/val", sess.graph)

            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("\n\n")

            for step in range(1, EPOCH + 1):
                _, summary, tra_loss, tra_acc = sess.run(
                    [train_op, merged_summary, cost, _accuracy],
                    feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                )

                # training
                if DO_SHOW and step % NUM_OF_SAVE_EPOCH == 0:
                    # write train curve on tensor board
                    train_writer.add_summary(summary, global_step=step)

                    val_summary, val_loss, val_acc = sess.run(
                        [merged_summary, cost, _accuracy],
                        feed_dict={self.tf_x: x_valid, self.tf_y: y_valid, self.keep_prob: KEEP_PROB}
                    )

                    # write validation curve on tensor board
                    val_writer.add_summary(val_summary, global_step=step)

                    print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc*100.0))
                    print("            valid loss =  %.5f, valid  acc = %.2f" % (val_loss, val_acc*100.0))

                    # save tensor every NUM_OF_SAVE_EPOCH
                    saver.save(sess, self.name_of_tensor + "model", global_step=step)

                    if self.__is_stopped_training(val_loss):
                        break

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
        # restore tensor
        self.__set_name_of_tensor()
        checkpoint = tf.train.get_checkpoint_state(self.name_of_tensor)
        paths = checkpoint.all_model_checkpoint_paths
        path = paths[len(paths) - (NUM_OF_LOSS_OVER_FIT + 1)]
        self.best_epoch = int(path.split("/")[-1].split("model-")[-1])
        self.num_of_dimension = len(x_test[0])

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)

            print("\n\n\nRead Neural Network -", self.name_of_tensor, "\n")

            # load tensor
            graph = tf.get_default_graph()
            tf_x = graph.get_tensor_by_name(NAME_X + ":0")
            tf_y = graph.get_tensor_by_name(NAME_Y + ":0")
            keep_prob = graph.get_tensor_by_name(NAME_PROB + ":0")
            hypothesis = graph.get_tensor_by_name(NAME_HYPO + ":0")
            predict = graph.get_tensor_by_name(NAME_PREDICT + ":0")
            num_of_hidden = graph.get_tensor_by_name(NAME_HIDDEN + ":0")
            learning_rate = graph.get_tensor_by_name(NAME_LEARNING_RATE + ":0")

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
        self.compute_score(__get_reverse(y_predict), __get_reverse(y_test), __get_reverse(h, is_hypothesis=True))
        self.set_score(target=KEY_IMMORTALITY)
        self.show_score(target=KEY_IMMORTALITY)
        self.set_plot(target=KEY_IMMORTALITY)

        # set score of mortality
        self.compute_score(y_predict, y_test, h)
        self.set_score(target=KEY_MORTALITY)
        self.show_score(target=KEY_MORTALITY)
        self.set_plot(target=KEY_MORTALITY)

        # set total score of immortality and mortality
        self.set_total_score()
        self.show_score(target=KEY_TOTAL)

    def save(self, data_handler):
        self.save_score(data_handler, self.best_epoch, self.num_of_dimension, self.num_of_hidden, self.learning_rate)
