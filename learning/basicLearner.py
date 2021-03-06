import tensorflow as tf
import math
from .neuralNetModel import TensorModel, EarlyStopping
from .variables import *


class NeuralNet(TensorModel):
    def __init__(self, is_cross_valid=True):
        super().__init__(is_cross_valid=is_cross_valid)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()
        self.early_stopping = EarlyStopping(patience=NUM_OF_LOSS_OVER_FIT, verbose=0)

    def __init_feed_forward_layer(self, num_of_input_nodes, num_of_output_nodes, input_layer):
        if NUM_HIDDEN_DIMENSION:
            num_hidden_node = NUM_HIDDEN_DIMENSION
        else:
            num_hidden_node = num_of_input_nodes

        tf_weight = list()
        tf_bias = list()
        tf_layer = [input_layer]

        # # make hidden layers
        for i in range(self.num_of_hidden):
            # set number of hidden node
            num_hidden_node = int(num_of_input_nodes / RATIO_HIDDEN)

            # append weight
            tf_weight.append(tf.get_variable(name="h_weight_" + str(i + 1) + '_' + str(self.num_of_fold),
                                             dtype=tf.float32,
                                             shape=[num_of_input_nodes, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            # append bias
            tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]),
                                       name="h_bias_" + str(i + 1) + '_' + str(self.num_of_fold)))
            layer = tf.add(tf.matmul(tf_layer[i], tf_weight[i]), tf_bias[i])

            # batch normalization
            bn_layer = tf.layers.batch_normalization(layer, training=True)

            # activate function
            hidden_layer = tf.nn.relu(bn_layer)

            # append layer and use dropout
            tf_layer.append(tf.nn.dropout(hidden_layer, keep_prob=self.keep_prob,
                                          name="dropout_" + str(i + 1) + '_' + str(self.num_of_fold)))

            # set number of node which is next layer
            num_of_input_nodes = int(num_of_input_nodes / RATIO_HIDDEN)

        tf_weight.append(tf.get_variable(dtype=tf.float32, shape=[num_hidden_node, 1],
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         name="o_weight_" + str(self.num_of_fold)))
        tf_bias.append(tf.Variable(tf.random_normal([num_of_output_nodes]),
                                   name="o_bias_" + str(self.num_of_fold)))

        if self.do_show:
            print("\n\n======== Feed Forward Layer ========")
            for i, layer in enumerate(tf_layer):
                print("Layer", i + 1, "-", layer.shape)

        # return X*W + b
        return tf.add(tf.matmul(tf_layer[-1], tf_weight[-1]), tf_bias[-1])

    def show_sets(self, y_train, y_valid, y_test):
        def __counting_mortality(_data):
            count = 0

            death_vector = [1]

            for _d in _data:
                if _d == death_vector:
                    count += 1

            return count

        if self.do_show:
            len_alive_train = len(y_train)
            len_alive_valid = len(y_valid)
            len_alive_test = len(y_test)

            len_death_train = __counting_mortality(y_train)
            len_death_valid = __counting_mortality(y_valid)
            len_death_test = __counting_mortality(y_test)

            print("\nAll   total count -", str(len_alive_train + len_alive_valid + len_alive_test).rjust(4),
                  "\tAlive count -", str(len_alive_train + len_alive_valid + len_alive_test -
                                         len_death_train - len_death_valid - len_death_test).rjust(4),
                  "\tDeath count -", str(len_death_train + len_death_valid + len_death_test).rjust(4))
            print("Train total count -", str(len_alive_train).rjust(4),
                  "\tAlive count -", str(len_alive_train - len_death_train).rjust(4),
                  "\tDeath count -", str(len_death_train).rjust(4))
            print("Valid total count -", str(len_alive_valid).rjust(4),
                  "\tAlive count -", str(len_alive_valid - len_death_valid).rjust(4),
                  "\tDeath count -", str(len_death_valid).rjust(4))
            print("Test  total count -", str(len_alive_test).rjust(4),
                  "\tAlive count -", str(len_alive_test - len_death_test).rjust(4),
                  "\tDeath count -", str(len_death_test).rjust(4), "\n\n")

    def training(self, x_train, y_train, x_valid, y_valid):
        self.init_place_holder(x_train, y_train)
        self.feed_forward(x_train, y_train, x_valid, y_valid, input_layer=self.tf_x)

    def init_place_holder(self, x_train, y_train):
        self.num_of_fold += 1
        self.num_of_input_nodes = len(x_train[0])
        self.num_of_output_nodes = len(y_train[0])
        self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_input_nodes],
                                   name=NAME_X + '_' + str(self.num_of_fold))
        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))
        # self.batch_prob = tf.placeholder(tf.bool, name=NAME_BATCH_PROB + '_' + str(self.num_of_fold))

    def feed_forward(self, x_train, y_train, x_valid, y_valid, input_layer):
        # initialize neural network
        hypothesis = self.__init_feed_forward_layer(num_of_input_nodes=self.num_of_input_nodes,
                                                    num_of_output_nodes=self.num_of_output_nodes,
                                                    input_layer=input_layer)
        h, y_predict, accuracy = self.__sess_run(hypothesis, x_train, y_train, x_valid, y_valid)
        self.compute_score(y_valid, y_predict, h, accuracy)

        if self.is_cross_valid:
            key = KEY_TEST
        else:
            key = KEY_VALID

        self.set_score(target=key)
        self.show_score(target=key)

    def __sess_run(self, hypothesis, x_data, y_data, x_test, y_test):
        if self.do_show:
            print("Layer O -", hypothesis.shape, "\n\n\n")

        num_of_class = self.num_of_output_nodes

        # Use softmax cross entropy
        if num_of_class > 1:
            with tf.name_scope(NAME_SCOPE_COST):
                cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.tf_y)
                cost = tf.reduce_mean(cost_i)
                cost_summ = tf.summary.scalar("cost", cost)

            with tf.name_scope(NAME_SCOPE_PREDICT):
                hypothesis = tf.nn.softmax(hypothesis)
                predict = tf.argmax(hypothesis, 1)
                correct_prediction = tf.equal(predict, tf.argmax(self.tf_y, 1))
                _accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
                accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

            train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Do not use softmax cross entropy
        else:
            with tf.name_scope("cost"):
                hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO + '_' + str(self.num_of_fold))
                cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
                cost_summ = tf.summary.scalar("cost", cost)

            with tf.name_scope("prediction"):
                predict = tf.cast(hypothesis > 0.5, dtype=tf.float32,
                                  name=NAME_PREDICT + '_' + str(self.num_of_fold))
                _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
                accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # split train, valid
        # x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data,
        #                                                       test_size=0.2,
        #                                                       random_state=SPLIT_SEED,
        #                                                       shuffle=True)

        x_train, x_valid, y_train, y_valid = self.train_valid_split(x_data=x_data, y_data=y_data)
        self.show_sets(y_train, y_valid, y_test)

        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()
        tf.Variable(self.learning_rate, name=NAME_LEARNING_RATE + '_' + str(self.num_of_fold))
        tf.Variable(self.num_of_hidden, name=NAME_HIDDEN + '_' + str(self.num_of_fold))

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            print("\n\n\n")

            train_writer = tf.summary.FileWriter(self.get_name_of_log() + "/train", sess.graph)
            val_writer = tf.summary.FileWriter(self.get_name_of_log() + "/val", sess.graph)
            saver = tf.train.Saver(max_to_keep=(NUM_OF_LOSS_OVER_FIT + 1))

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            batch_iter = int(math.ceil(len(x_train) / BATCH_SIZE))

            for step in range(1, self.best_epoch + 1):
                # mini-batch
                for i in range(batch_iter):
                    batch_x = x_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                    batch_y = y_train[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

                    _, tra_loss = sess.run(
                        [train_op, cost],
                        feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.keep_prob: KEEP_PROB}
                    )

                train_summary, tra_loss, tra_acc = sess.run(
                    [merged_summary, cost, _accuracy],
                    feed_dict={self.tf_x: x_train, self.tf_y: y_train, self.keep_prob: KEEP_PROB}
                )

                train_writer.add_summary(train_summary, global_step=step)

                val_summary, val_loss, val_acc = sess.run(
                    [merged_summary, cost, _accuracy],
                    feed_dict={self.tf_x: x_valid, self.tf_y: y_valid, self.keep_prob: KEEP_PROB}
                )

                # write validation curve on tensor board
                val_writer.add_summary(val_summary, global_step=step)

                saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                if self.early_stopping.validate(val_loss, val_acc):
                    self.__show_loss(step, tra_loss, val_loss, tra_acc, val_acc, do_show=True)
                    break

                self.__show_loss(step, tra_loss, val_loss, tra_acc, val_acc)

            h, p, acc = sess.run([hypothesis, predict, _accuracy],
                                 feed_dict={self.tf_x: x_test, self.tf_y: y_test, self.keep_prob: 1.0})

        tf.reset_default_graph()

        return h, p, acc

    def __show_loss(self, step, tra_loss, val_loss, tra_acc, val_acc, do_show=False):
        if self.do_show and step % NUM_OF_SHOW_EPOCH == 0:
            do_show = True

        if do_show:
            print("Step %5d, train loss =  %.5f, train  acc = %.2f" % (step, tra_loss, tra_acc * 100.0))
            print("            valid loss =  %.5f, valid  acc = %.2f" % (val_loss, val_acc * 100.0))

            if math.isnan(tra_loss) or math.isnan(val_loss):
                print("Vanishing weights!!\n\n")
                exit(-1)

    def __is_stopped_training(self, val_loss):
        self.val_loss_list.append(val_loss)

        cnt_train = len(self.val_loss_list)

        if cnt_train < 5:
            return False
        else:
            loss_default = self.val_loss_list[cnt_train - (NUM_OF_LOSS_OVER_FIT + 1)]
            cnt_loss_over_fit = int()

            for loss in self.val_loss_list[cnt_train - NUM_OF_LOSS_OVER_FIT:cnt_train]:
                if loss > loss_default:
                    cnt_loss_over_fit += 1
                # loss_default = loss

            if cnt_loss_over_fit == NUM_OF_LOSS_OVER_FIT:
                self.val_loss_list.clear()
                return True
            else:
                return False

    def load_nn(self, x_test, y_test):
        self.num_of_fold += 1
        checkpoint = tf.train.get_checkpoint_state(self.get_name_of_tensor())
        paths = checkpoint.all_model_checkpoint_paths

        # if self.is_cross_valid:
        #     path = paths[-1]
        # else:
        path = paths[len(paths) - (NUM_OF_LOSS_OVER_FIT + 1)]

        self.best_epoch_list.append(int(path.split("/")[-1].split("model-")[-1]))
        self.num_of_dimension = len(x_test[0])

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(sess, path)

            print("\n\n\ncheckpoint -", path, "\nBest Epoch -", self.best_epoch_list[-1], "\n")

            # load tensor
            graph = tf.get_default_graph()
            str_n_fold = str(self.num_of_fold)
            tf_x = graph.get_tensor_by_name(NAME_X + "_" + str_n_fold + ":0")
            tf_y = graph.get_tensor_by_name(NAME_Y + "_" + str_n_fold + ":0")
            keep_prob = graph.get_tensor_by_name(NAME_PROB + "_" + str_n_fold + ":0")
            hypothesis = graph.get_tensor_by_name(NAME_SCOPE_COST + "/" + NAME_HYPO + "_" + str_n_fold + ":0")
            predict = graph.get_tensor_by_name(NAME_SCOPE_PREDICT + "/" + NAME_PREDICT + "_" + str_n_fold + ":0")
            num_of_hidden = graph.get_tensor_by_name(NAME_HIDDEN + "_" + str_n_fold + ":0")
            learning_rate = graph.get_tensor_by_name(NAME_LEARNING_RATE + "_" + str_n_fold + ":0")

            self.num_of_hidden, self.learning_rate = sess.run([num_of_hidden, learning_rate])
            h, y_predict = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test, keep_prob: 1})

        tf.reset_default_graph()

        return h, y_predict

    def save(self, data_handler=False):
        # set total score of immortality and mortality

        self.set_performance()
        self.show_performance()

        if self.is_cross_valid:
            self.save_score_cross_valid(best_epoch=self.best_epoch_list,
                                        num_of_dimension=self.num_of_dimension,
                                        num_of_hidden=self.num_of_hidden,
                                        learning_rate=self.learning_rate)
        else:
            self.save_score(data_handler,
                            best_epoch=self.best_epoch,
                            num_of_dimension=self.num_of_dimension,
                            num_of_hidden=self.num_of_hidden,
                            learning_rate=self.learning_rate)


class ConvolutionNet(NeuralNet):
    def __init__(self, is_cross_valid=True):
        super().__init__(is_cross_valid=is_cross_valid)

    def training(self, x_train, y_train, x_valid, y_valid, train_ct_image=False):
        self.init_place_holder(x_train, y_train)

        convolution_layer, num_of_dimension = self.__init_convolution_model(self.num_of_input_nodes)

        self.num_of_input_nodes = num_of_dimension
        self.feed_forward(x_train, y_train, x_valid, y_valid, input_layer=convolution_layer)

    def __init_convolution_model(self, num_of_input_nodes):
        num_of_image = int(math.sqrt(num_of_input_nodes))
        num_of_filter = [20, 50, 100, 200]
        size_of_filter = [5, 5, 3, 3]
        tf_x_img = tf.reshape(self.tf_x, [-1, num_of_image, num_of_image, 1])

        # 5 x 5 x 1 x 20
        filter_1 = tf.Variable(
            tf.random_normal([size_of_filter[0], size_of_filter[0], 1, num_of_filter[0]], stddev=0.01),
            name="cnn_filter_1")
        conv_1 = tf.nn.conv2d(tf_x_img, filter_1, strides=[1, 1, 1, 1], padding="SAME",
                              name="conv_1_" + str(self.num_of_fold))
        bn_1 = tf.layers.batch_normalization(conv_1, training=True, name="bn_1_" + str(self.num_of_fold))
        relu_1 = tf.nn.relu(bn_1, name="relu_1_" + str(self.num_of_fold))
        pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name="pool_1_" + str(self.num_of_fold))
        dropout_1 = tf.nn.dropout(pool_1, keep_prob=self.keep_prob, name="dropout_1_" + str(self.num_of_fold))

        # 5 x 5 x 20 x 50
        filter_2 = tf.Variable(
            tf.random_normal([size_of_filter[1], size_of_filter[1], num_of_filter[0], num_of_filter[1]], stddev=0.01),
            name="cnn_filter_2")
        conv_2 = tf.nn.conv2d(dropout_1, filter_2, strides=[1, 1, 1, 1], padding="SAME",
                              name="conv_2_" + str(self.num_of_fold))
        bn_2 = tf.layers.batch_normalization(conv_2, training=True, name="bn_2_" + str(self.num_of_fold))
        relu_2 = tf.nn.relu(bn_2, name="relu_2_" + str(self.num_of_fold))
        pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name="pool_2_" + str(self.num_of_fold))
        dropout_2 = tf.nn.dropout(pool_2, keep_prob=self.keep_prob, name="dropout_2_" + str(self.num_of_fold))

        # 3 x 3 x 50 x 100
        filter_3 = tf.Variable(
            tf.random_normal([size_of_filter[2], size_of_filter[2], num_of_filter[1], num_of_filter[2]], stddev=0.01),
            name="cnn_filter_3")
        conv_3 = tf.nn.conv2d(dropout_2, filter_3, strides=[1, 1, 1, 1], padding="SAME",
                              name="conv_3_" + str(self.num_of_fold))
        bn_3 = tf.layers.batch_normalization(conv_3, training=True, name="bn_3_" + str(self.num_of_fold))
        relu_3 = tf.nn.relu(bn_3, name="relu_3_" + str(self.num_of_fold))
        pool_3 = tf.nn.max_pool(relu_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name="pool_3_" + str(self.num_of_fold))
        dropout_3 = tf.nn.dropout(pool_3, keep_prob=self.keep_prob, name="dropout_3_" + str(self.num_of_fold))

        # 3 x 3 x 100 x 200
        filter_4 = tf.Variable(
            tf.random_normal([size_of_filter[3], size_of_filter[3], num_of_filter[2], num_of_filter[3]], stddev=0.01),
            name="cnn_filter_4")
        conv_4 = tf.nn.conv2d(pool_3, filter_4, strides=[1, 1, 1, 1], padding="SAME",
                              name="conv_4_" + str(self.num_of_fold))
        bn_4 = tf.layers.batch_normalization(conv_4, training=True, name="bn_4_" + str(self.num_of_fold))
        relu_4 = tf.nn.relu(bn_4, name="relu_4_" + str(self.num_of_fold))
        pool_4 = tf.nn.max_pool(relu_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name="pool_4_" + str(self.num_of_fold))
        dropout_4 = tf.nn.dropout(pool_4, keep_prob=self.keep_prob, name="dropout_4_" + str(self.num_of_fold))

        fc_nodes = dropout_4.shape
        fc_nodes = int(fc_nodes[1] * fc_nodes[2] * fc_nodes[3])
        convolution_layer = tf.reshape(pool_4, [-1, fc_nodes], name="cnn_span_layer_" + str(self.num_of_fold))

        if self.do_show:
            print("\n\n======== Convolution Layer ========")
            print("tf_x     -", self.tf_x.shape)
            print("tf_x_img -", tf_x_img.shape)

            print("\nconv_1 -", conv_1.shape)
            print("pool_1 -", pool_1.shape)
            print("\nconv_2 -", conv_2.shape)
            print("pool_2 -", pool_2.shape)
            print("\nconv_3 -", conv_3.shape)
            print("pool_3 -", pool_3.shape)
            print("\nconv_4 -", conv_4.shape)
            print("pool_4 -", pool_4.shape)
            print("\ncnn_span_layer -", convolution_layer.shape)

        return convolution_layer, fc_nodes
