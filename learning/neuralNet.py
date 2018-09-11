import tensorflow as tf
import DeepPredict.arguments as op
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .plot import MyPlot
import os
import shutil


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
            self.my_plot.plot(fpr, tpr, alpha=0.3, label='ROC %d (AUC = %0.1f)' % (k_fold+1, self.score["AUC"][-1]))

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

    def __init_multi_layer(self, num_input_node):
        if NUM_HIDDEN_DIMENSION:
            num_hidden_node = NUM_HIDDEN_DIMENSION
        else:
            num_hidden_node = num_input_node

        tf_weight = list()
        tf_bias = list()
        tf_layer = [self.tf_x]
        if op.DO_SHOW:
            print("\n\n--- Layer Information ---")
            print(tf_layer[0].shape)

        # # make hidden layers
        for i in range(op.NUM_HIDDEN_LAYER):
            # set number of hidden node
            num_hidden_node = int(num_input_node / RATIO_HIDDEN)

            # append weight
            tf_weight.append(tf.get_variable("h_weight_" + str(i + 1), dtype=tf.float32,
                                             shape=[num_input_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            # append bias
            tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i + 1)))
            layer = tf.add(tf.matmul(tf_layer[i], tf_weight[i]), tf_bias[i])
            if op.DO_SHOW:
                print(layer.shape)

            # append hidden layer
            hidden_layer = tf.nn.relu(layer)
            tf_layer.append(tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob, name="dropout_" + str(i + 1)))

            # set number of node which is next layer
            num_input_node = int(num_input_node / RATIO_HIDDEN)

        tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                         initializer=tf.contrib.layers.xavier_initializer()))
        tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

        # return X*W + b
        return tf.add(tf.matmul(tf_layer[-1], tf_weight[-1]), tf_bias[-1])

    def feed_forward_nn(self, k_fold, x_train, y_train, x_test, y_test):
        save_dir = self.__init_save_dir(k_fold)
        num_input_node = len(x_train[0])

        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_input_node], name=NAME_X)
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)
        self.keep_prob = tf.placeholder(tf.float32)

        # make hidden layers
        hypothesis = self.__init_multi_layer(num_input_node=num_input_node)

        if op.DO_SHOW:
            print(hypothesis.shape)
            print("\n")
        hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO)

        with tf.name_scope("cost"):
            cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=tf_y))
            cost_summ = tf.summary.scalar("cost", cost)

        # train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        # cut off
        predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
        _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.__init_log_file_name(k_fold))
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

        _precision = precision_score(y_test, p)
        _recall = recall_score(y_test, p)
        _f1 = f1_score(y_test, p)
        _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
        _logistic_fpr *= 100
        _logistic_tpr *= 100
        _auc = auc(_logistic_fpr, _logistic_tpr) / 100

        if _precision == 0 or _recall == 0:
            print("\n\n------------\nIt's not working")
            print('k-fold : %d, Precision : %.1f, Recall : %.1f' % (k_fold + 1, (_precision * 100), (_recall * 100)))
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
            self.my_plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3,
                              label='ROC %d (AUC = %0.1f)' % (k_fold + 1, _auc))

    def load_feed_forward_nn(self, k_fold, x_test, y_test):
        tensor_load = self.__load_tensor(k_fold)

        sess = tf.Session()
        saver = tf.train.import_meta_graph(tensor_load + 'model-' + str(op.EPOCH) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(tensor_load))

        print("\n\n\nRead Neural Network -", tensor_load, "\n")
        graph = tf.get_default_graph()
        tf_x = graph.get_tensor_by_name(NAME_X + ":0")
        tf_y = graph.get_tensor_by_name(NAME_Y + ":0")
        hypothesis = graph.get_tensor_by_name(NAME_HYPO + ":0")
        predict = graph.get_tensor_by_name(NAME_PREDICT + ":0")

        h, p = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test})

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
