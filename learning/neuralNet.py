import tensorflow as tf
import DMP.utils.arg_training as op
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .plot import MyPlot
import os
import shutil
import math


class MyNeuralNetwork(MyPlot):
    def __init__(self):
        super().__init__()
        self.__score = {
            "P": list(),
            "R": list(),
            "F1": list(),
            "Acc": list(),
            "AUC": list()
        }
        self.tf_x = None
        self.tf_y = None
        self.keep_prob = None

    @property
    def score(self):
        return self.__score

    def add_score(self, **kwargs):
        for k, v in kwargs.items():
            self.__score[k].append(v)

    def show_score(self, k_fold, fpr, tpr):
        if op.DO_SHOW:
            print('\n\n')
            print(k_fold + 1, "fold")
            print('Precision : %.1f' % (self.score["P"][-1] * 100))
            print('Recall    : %.1f' % (self.score["R"][-1] * 100))
            print('F1-Score  : %.1f' % (self.score["F1"][-1] * 100))
            print('Accuracy  : %.1f' % (self.score["Acc"][-1] * 100))
            print('AUC       : %.1f' % self.score["AUC"][-1])
            # self.my_plot.plot(fpr, tpr, alpha=0.3, label='ROC %d (AUC = %0.1f)' % (k_fold+1, self.score["AUC"][-1]))

    def show_total_score(self, _method):
        print("\n\n============ " + _method + " ============\n")
        print("Total precision - %.1f" % ((sum(self.score["P"]) / op.NUM_FOLDS) * 100))
        print("Total recall    - %.1f" % ((sum(self.score["R"]) / op.NUM_FOLDS) * 100))
        print("Total F1-Score  - %.1f" % ((sum(self.score["F1"]) / op.NUM_FOLDS) * 100))
        print("Total accuracy  - %.1f" % ((sum(self.score["Acc"]) / op.NUM_FOLDS) * 100))
        print("Total auc       - %.1f" % (sum(self.score["AUC"]) / op.NUM_FOLDS))
        print("\n\n======================================\n")

    @staticmethod
    def __init_log_file_name(k_fold):
        log_name = "./logs/" + op.SAVE_DIR_NAME + op.USE_ID + "log_"

        if op.NUM_HIDDEN_LAYER < 10:
            log_name += "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            log_name += "h_" + str(op.NUM_HIDDEN_LAYER)

        log_name += "_ep_" + str(op.EPOCH) + "_k_" + str(k_fold + 1)

        if op.USE_W2V:
            log_name += "_w2v"

        if os.path.isdir(log_name):
            shutil.rmtree(log_name)

        return log_name

    @staticmethod
    def __init_save_dir(k_fold):
        if op.NUM_HIDDEN_LAYER < 10:
            _hidden_ = "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            _hidden_ = "h_" + str(op.NUM_HIDDEN_LAYER)

        _epoch_ = "_ep_" + str(op.EPOCH) + "_"

        _save_dir = TENSOR_PATH + op.SAVE_DIR_NAME

        if not os.path.isdir(_save_dir):
            os.mkdir(_save_dir)

        _save_dir += _hidden_ + _epoch_ + str(k_fold + 1) + "/"

        if os.path.isdir(_save_dir):
            shutil.rmtree(_save_dir)
        os.mkdir(_save_dir)

        return _save_dir

    @staticmethod
    def __load_tensor(k_fold):
        if op.NUM_HIDDEN_LAYER < 10:
            _hidden_ = "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            _hidden_ = "h_" + str(op.NUM_HIDDEN_LAYER)

        _epoch_ = "_ep_" + str(op.EPOCH) + "_"
        _save_dir = TENSOR_PATH + op.SAVE_DIR_NAME

        if op.USE_ID:
            tensor_load = _save_dir + op.USE_ID.split('#')[0] + "_"
        else:
            tensor_load = _save_dir

        tensor_load += _hidden_ + _epoch_ + str(k_fold + 1) + "/"

        return tensor_load

    def __init_feed_forward_layer(self, num_input_node, input_layer):
        if NUM_HIDDEN_DIMENSION:
            num_hidden_node = NUM_HIDDEN_DIMENSION
        else:
            num_hidden_node = num_input_node

        tf_weight = list()
        tf_bias = list()
        tf_layer = [input_layer]

        # # make hidden layers
        for i in range(op.NUM_HIDDEN_LAYER):
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

        if op.DO_SHOW:
            print("\n\n--- Feed Forward Layer Information ---")
            for i, layer in enumerate(tf_layer):
                print("Layer", i + 1, "-", layer.shape)

        # return X*W + b
        return tf.add(tf.matmul(tf_layer[-1], tf_weight[-1]), tf_bias[-1])

    def feed_forward_nn(self, k_fold, x_train, y_train, x_test, y_test):
        save_dir = self.__init_save_dir(k_fold)
        log_dir = self.__init_log_file_name(k_fold)
        num_of_dimension = len(x_train[0])

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension], name=NAME_X)
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB)

        # initialize neural network
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=self.tf_x)
        h, p, acc = self.__sess_run(hypothesis, x_train, y_train, x_test, y_test, log_dir, save_dir)
        self.__compute_score(k_fold, y_test, h, p, acc)

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

        if op.DO_SHOW:
            print("\n\n--- Convolution Layer Information ---")
            print("tf_x     -", self.tf_x.shape)
            print("tf_x_img -", tf_x_img.shape)

            print("\n\nfilter_1 -", filter_1.shape)
            print(" layer_1 -", layer_1.shape)
            print("\n\nfilter_2 -", filter_2.shape)
            print(" layer_2 -", layer_2.shape)

        return convolution_layer, num_of_dimension

    def convolution_nn(self, k_fold, x_train, y_train, x_test, y_test):
        log_dir = self.__init_log_file_name(k_fold)
        save_dir = self.__init_save_dir(k_fold)
        num_of_dimension = len(x_train[0])

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_of_dimension], name=NAME_X)
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB)

        # concat CNN to Feed Forward NN
        convolution_layer, num_of_dimension = self.__init_convolution_layer(num_of_dimension)
        hypothesis = self.__init_feed_forward_layer(num_input_node=num_of_dimension, input_layer=convolution_layer)
        h, p, acc = self.__sess_run(hypothesis, x_train, y_train, x_test, y_test, log_dir, save_dir)
        self.__compute_score(k_fold, y_test, h, p, acc)

    def __sess_run(self, hypothesis, x_train, y_train, x_test, y_test, log_dir, save_dir):
        if op.DO_SHOW:
            print("Layer O -", hypothesis.shape)
            print("\n")
        hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO)

        with tf.name_scope("cost"):
            cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=tf_y))
            cost_summ = tf.summary.scalar("cost", cost)

        train_op = tf.train.AdamOptimizer(learning_rate=op.LEARNING_RATE).minimize(cost)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=op.LEARNING_RATE).minimize(cost)

        # cut off
        predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
        _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(sess.graph)  # Show the graph

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # if self.is_closed:
            for step in range(op.EPOCH + 1):
                summary, cost_val, _ = sess.run([merged_summary, cost, train_op],
                                                feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: 0.7})
                writer.add_summary(summary, global_step=step)

                if op.DO_SHOW and step % (op.EPOCH / 10) == 0:
                    print(str(step).rjust(5), cost_val)

            h, p, acc = sess.run([hypothesis, predict, _accuracy],
                                 feed_dict={self.tf_x: x_test, self.tf_y: y_test, self.keep_prob: 1})

            saver.save(sess, save_dir + "model", global_step=op.EPOCH)

        tf.reset_default_graph()

        return h, p, acc

    def __compute_score(self, k_fold, y_test, h, p, acc):
        _precision = precision_score(y_test, p)
        _recall = recall_score(y_test, p)
        _f1 = f1_score(y_test, p)

        try:
            _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
        except ValueError:
            print("\n\nWhere fold -", k_fold, " cost is NaN !!")
            # print("erase  log directory -", log_dir)
            # print("erase save directory -", save_dir)
            # shutil.rmtree(save_dir)
            # shutil.rmtree(log_dir)
            # os.rmdir(save_dir)
            # os.rmdir(log_dir)
            exit(-1)
        else:
            _logistic_fpr *= 100
            _logistic_tpr *= 100
            _auc = auc(_logistic_fpr, _logistic_tpr) / 100

            if _precision == 0 or _recall == 0:
                print("\n\n------------\nIt's not working")
                print('k-fold : %d, Precision : %.1f, Recall : %.1f' %
                      (k_fold + 1, (_precision * 100), (_recall * 100)))
                print("\n------------")

            self.add_score(**{"P": _precision, "R": _recall, "F1": _f1, "Acc": acc, "AUC": _auc})

            if op.DO_SHOW:
                print('\n\n')
                print(k_fold + 1, "fold")
                print('Precision : %.1f' % (_precision * 100))
                print('Recall    : %.1f' % (_recall * 100))
                print('F1-Score  : %.1f' % (_f1 * 100))
                print('Accuracy  : %.1f' % (acc * 100))
                print('AUC       : %.1f' % _auc)
                # self.my_plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3,
                #                   label='ROC %d (AUC = %0.1f)' % (k_fold + 1, _auc))

    def load_feed_forward_nn(self, k_fold, x_test, y_test):
        tensor_load = self.__load_tensor(k_fold)

        sess = tf.Session()
        saver = tf.train.import_meta_graph(tensor_load + 'model-' + str(op.EPOCH) + '.meta')
        saver.restore(sess, tensor_load + 'model-' + str(op.EPOCH))

        print("\n\n\nRead Neural Network -", tensor_load, "\n")

        graph = tf.get_default_graph()
        tf_x = graph.get_tensor_by_name(NAME_X + ":0")
        tf_y = graph.get_tensor_by_name(NAME_Y + ":0")
        hypothesis = graph.get_tensor_by_name(NAME_HYPO + ":0")
        predict = graph.get_tensor_by_name(NAME_PREDICT + ":0")
        keep_prob = graph.get_tensor_by_name(NAME_PROB + ":0")

        h, p = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test, keep_prob: 1})

        logistic_fpr, logistic_tpr, _ = roc_curve(y_test, h)
        logistic_fpr *= 100
        logistic_tpr *= 100

        _precision = precision_score(y_test, p)
        _recall = recall_score(y_test, p)
        _f1 = f1_score(y_test, p)
        _accuracy = accuracy_score(y_test, p)
        _auc = auc(logistic_fpr, logistic_tpr) / 100

        self.add_score(**{"P": _precision, "R": _recall, "F1": _f1, "Acc": _accuracy, "AUC": _auc})
        self.show_score(k_fold, fpr=logistic_fpr, tpr=logistic_tpr)