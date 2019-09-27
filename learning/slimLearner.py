from DMP.learning.neuralNet import TensorModel
from DMP.modeling.tfRecorder import *
from .variables import *
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.slim.nets
import sys
from os import path, getcwd
from .neuralNet import EarlyStopping

SLIM_PATH = path.dirname(path.abspath(getcwd())) + '/models/research/slim'
sys.path.append(SLIM_PATH)

VGG_PATH = 'dataset/images/ckpt/vgg_16.ckpt'
NUM_OF_EARLY_STOPPING = 5


class SlimLearner(TensorModel):
    def __init__(self, model=None, tf_name_vector=None):
        super().__init__(is_cross_valid=False)
        # self.tf_recorder = TfRecorder(self.tf_record_path)
        self.num_of_input_nodes = self.tf_recorder.log[KEY_OF_TRAIN + KEY_OF_DIM]
        self.num_of_output_nodes = 1
        self.tf_name_vector = tf_name_vector
        self.is_cross_valid = self.tf_recorder.is_cross_valid
        self.model = model

        if self.model == "ffnn":
            self.shape = None
        else:
            # self.shape = (None, Width, Height, channels)
            shape = self.tf_recorder.log[KEY_OF_SHAPE][:]
            shape.insert(0, None)
            self.shape = shape

        self.early_stopping = EarlyStopping(patience=NUM_OF_EARLY_STOPPING, verbose=1, minimum_epoch=30)

        self.loss_dict = {
            "train": list(),
            "valid": list()
        }
        self.acc_dict = {
            "train": list(),
            "valid": list()
        }

        if self.do_show:
            if self.is_cross_valid:
                print('5-fold cross validation')
            else:
                print('hyper-parameter optimization')
                if USE_EARLY_STOPPING:
                    print("Use early stopping")
                else:
                    print("Do not use early stopping")

            if self.model == "tuning" or self.model == "full":
                print('Load  Vgg16  Model -', self.model, '\nis_training option - True\n\n\n')
            elif self.model == "transfer":
                print('Load  Vgg16  Model -', self.model, '\nis_training option - False\n\n\n')
            else:
                print("Feed Forward Neural Net\n\n\n")

    def __init_var_result(self):
        self.h = list()
        self.p = list()
        self.y_test = list()

    def run_fine_tuning(self):
        self.num_of_fold += 1

        # fine tuning
        if self.shape:
            self.tf_x = tf.placeholder(dtype=tf.float32, shape=self.shape,
                                       name=NAME_X + '_' + str(self.num_of_fold))
        else:
            self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_input_nodes],
                                       name=NAME_X + '_' + str(self.num_of_fold))

        self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_of_output_nodes],
                                   name=NAME_Y + '_' + str(self.num_of_fold))
        self.keep_prob = tf.placeholder(tf.float32, name=NAME_PROB + '_' + str(self.num_of_fold))

        self.__init_var_result()

        # fine tuning
        if self.shape:
            self.__fine_tuning()
        else:
            self.__training()

        self.compute_score(self.y_test, self.p, self.h)
        self.set_score(target=KEY_TEST)
        self.show_score(target=KEY_TEST)

    def __init_pre_trained_model(self):
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope()):
            # If the model is transfer learning, parameter will be not trained during training
            if self.model == "transfer":
                is_training = False
            # If the model is fine-tuning, parameter will be trained during training
            else:
                is_training = True
            logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1, is_training=is_training)
            # logits, end_points = vgg.vgg_16(inputs=self.tf_x, num_classes=1000, is_training=is_training)

            return logits, end_points

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

            # append hidden layer
            hidden_layer = tf.nn.relu(layer)
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

    def __fine_tuning(self):
        logits, end_points = self.__init_pre_trained_model()
        exclude = ['vgg_16/fc8']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        fc_7 = end_points['vgg_16/fc7']

        str_n_fold = '_' + str(self.num_of_fold)

        W = tf.Variable(tf.random_normal([4096, 1], mean=0.0, stddev=0.02), name=NAME_FC_W + str_n_fold)
        b = tf.Variable(tf.random_normal([1], mean=0.0), name=NAME_FC_B + str_n_fold)

        fc = tf.reshape(fc_7, [-1, W.get_shape().as_list()[0]], name=NAME_FC + str_n_fold)
        logitx = tf.nn.bias_add(tf.matmul(fc, W), b)

        with tf.name_scope(NAME_SCOPE_COST):
            hypothesis = tf.nn.sigmoid(logitx, name=NAME_HYPO + '_' + str(self.num_of_fold))
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx, labels=self.tf_y))
            cost_summary = tf.summary.scalar("cost", cost)

        with tf.name_scope(NAME_SCOPE_PREDICT):
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT + str_n_fold)
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", acc)

        if self.model == "transfer":
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, var_list=[W, b])
        else:
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, variables_to_restore)
        self.__sess_run(hypothesis, train_step, cost, acc, init_fn)

    # def __fine_tuning(self):
    #     logits, end_points = self.__init_pre_trained_model()
    #     exclude = ['vgg_16/fc8']
    #     variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    #     init_fn = slim.assign_from_checkpoint_fn(VGG_PATH, variables_to_restore)
    #
    #     with tf.name_scope(NAME_SCOPE_COST):
    #         hypothesis = tf.nn.sigmoid(logits, name=NAME_HYPO + '_' + str(self.num_of_fold))
    #         cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.tf_y))
    #         cost_summary = tf.summary.scalar("cost", cost)
    #
    #     with tf.name_scope(NAME_SCOPE_PREDICT):
    #         predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT + '_' + str(self.num_of_fold))
    #         acc = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
    #         accuracy_summary = tf.summary.scalar("accuracy", acc)
    #
    #     if self.model == "transfer":
    #         train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
    #     else:
    #         train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
    #
    #     self.__sess_run(hypothesis, train_step, cost, acc, init_fn)

    @staticmethod
    def __show_params(sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            # print(v)
            print()

    def __training(self):
        hypothesis = self.__init_feed_forward_layer(num_of_input_nodes=self.num_of_input_nodes,
                                                    num_of_output_nodes=self.num_of_output_nodes,
                                                    input_layer=self.tf_x)

        if self.do_show:
            print("Layer O -", hypothesis.shape, "\n\n\n")

        with tf.name_scope("cost"):
            hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO + '_' + str(self.num_of_fold))
            cost = -tf.reduce_mean(self.tf_y * tf.log(hypothesis) + (1 - self.tf_y) * tf.log(1 - hypothesis))
            cost_summ = tf.summary.scalar("cost", cost)

        with tf.name_scope("prediction"):
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32,
                              name=NAME_PREDICT + '_' + str(self.num_of_fold))
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, self.tf_y), dtype=tf.float32))
            accuracy_summ = tf.summary.scalar("accuracy", acc)

        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        self.__sess_run(hypothesis, train_step, cost, acc)

    def __sess_run(self, hypothesis, train_step, cost, acc, init_fn=None):
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

            # saver = tf.train.Saver(max_to_keep=(NUM_OF_EARLY_STOPPING + 1))
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "train", sess.graph)
            valid_writer = tf.summary.FileWriter(self.name_of_log + "valid", sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if init_fn:
                init_fn(sess)

            step = int()
            try:
                n_iter = int()
                batch_iter = int(self.tf_recorder.log[KEY_OF_TRAIN] / BATCH_SIZE) + 1

                # Training scope
                while not coord.should_stop():
                    n_iter += 1
                    x_batch, y_batch, x_img, x_name = sess.run(next_train_element)
                    # x_array = list()

                    # early stop for avoid over-fitting
                    # if not self.early_stopping.is_stop:
                    if self.shape:
                        target = x_img
                    else:
                        target = x_batch

                        # for name, x in zip(x_name, x_batch):
                        #     x_array.append(self.tf_name_vector[name.decode('utf-8')][0])
                        #
                        # target = np.array(x_array)

                    train_summary, _, tra_loss, tra_acc = sess.run(
                        [merged_summary, train_step, cost, acc],
                        feed_dict={self.tf_x: target, self.tf_y: y_batch, self.keep_prob: KEEP_PROB}
                    )
                    train_writer.add_summary(train_summary, global_step=n_iter)

                    self.loss_dict["train"].append(tra_loss)
                    self.acc_dict["train"].append(tra_acc)

                    # epoch
                    if n_iter % batch_iter == 0:
                        step += 1
                        self.__set_valid_loss(sess, n_iter, iterator_valid, merged_summary, cost, acc, valid_writer)
                        # saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")
                        if self.__set_average_values(step):
                            coord.request_stop()
            except tf.errors.OutOfRangeError:
                print("tfErrors]OutOfRangeError\n\n")
                exit(-1)
            finally:
                self.save_loss_plot(log_path=self.name_of_log, step_list=[step for step in range(1, step + 1)])
                saver = tf.train.Saver()
                saver.save(sess, global_step=step, save_path=self.get_name_of_tensor() + "/model")

                # set self.h, self.p, self.y_test
                self.__set_test_prob(sess, iterator_test, hypothesis)

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

                    if self.shape:
                        target = x_valid_img
                    else:
                        target = x_valid_batch

                    valid_summary, val_loss, val_acc = sess.run(
                        [merged_summary, cost, accuracy],
                        feed_dict={self.tf_x: target, self.tf_y: y_valid_batch,
                                   self.keep_prob: KEEP_PROB}
                    )
                    valid_writer.add_summary(valid_summary, global_step=n_iter)
                    self.loss_dict["valid"].append(val_loss)
                    self.acc_dict["valid"].append(val_acc)
            except tf.errors.OutOfRangeError:
                pass

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

    def __set_test_prob(self, sess, iterator, hypothesis):
        h_list = list()
        y_test = list()

        # Test scope
        sess.run(iterator.initializer)
        next_element = iterator.get_next()
        try:
            while True:
                x_batch, y_batch, x_img, tensor_name = sess.run(next_element)

                if self.shape:
                    target = x_img
                else:
                    target = x_batch

                h_batch = sess.run(hypothesis, feed_dict={self.tf_x: target, self.tf_y: y_batch, self.keep_prob: 1})

                for h, y in zip(h_batch, y_batch):
                    h_list.append(h)
                    y_test.append(y)
        except tf.errors.OutOfRangeError:
            self.h = np.array(h_list)
            self.p = (self.h > 0.5)
            self.y_test = np.array(y_test)

    def load_nn(self):
        self.num_of_fold += 1
        checkpoint = tf.train.get_checkpoint_state(self.get_name_of_tensor())
        paths = checkpoint.all_model_checkpoint_paths
        target_path = paths[-1]
        # target_path = paths[len(paths) - (NUM_OF_EARLY_STOPPING + 1)]

        self.best_epoch = int(target_path.split("/")[-1].split("model-")[-1])
        self.num_of_dimension = self.tf_recorder.log[KEY_OF_TRAIN + KEY_OF_DIM]
        self.__init_var_result()

        tf_test_record = self.init_tf_record_tensor(key=KEY_OF_TEST, is_test=True)
        iterator_test = tf_test_record.make_initializable_iterator()

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(target_path + '.meta')
            saver.restore(sess, target_path)

            print("\n\n\ncheckpoint -", target_path, "\nBest Epoch -", self.best_epoch, "\n")

            # load tensor
            graph = tf.get_default_graph()
            str_n_fold = str(self.num_of_fold)

            self.tf_x = graph.get_tensor_by_name(NAME_X + "_" + str_n_fold + ":0")
            self.tf_y = graph.get_tensor_by_name(NAME_Y + "_" + str_n_fold + ":0")
            self.keep_prob = graph.get_tensor_by_name(NAME_PROB + "_" + str_n_fold + ":0")

            # get num_of_hidden, learning_rate
            num_of_hidden = graph.get_tensor_by_name(NAME_HIDDEN + "_" + str_n_fold + ":0")
            learning_rate = graph.get_tensor_by_name(NAME_LEARNING_RATE + "_" + str_n_fold + ":0")
            self.num_of_hidden, self.learning_rate = sess.run([num_of_hidden, learning_rate])

            # self.w = graph.get_tensor_by_name(NAME_FC_W + '_' + str(self.num_of_fold) + ":0")
            #     b = graph.get_tensor_by_name(NAME_FC_B + '_' + str(self.num_of_fold) +  ":0")

            hypothesis = graph.get_tensor_by_name(NAME_SCOPE_COST + "/" + NAME_HYPO + "_" + str_n_fold + ":0")
            # predict = graph.get_tensor_by_name(NAME_SCOPE_PREDICT + "/" + NAME_PREDICT + "_" + str_n_fold + ":0")

            self.__set_test_prob(sess, iterator_test, hypothesis)

        tf.reset_default_graph()
        self.clear_tensor()

    def save(self):
        class Handler:
            def __init__(self, tf_recorder):
                self.count_all = [
                    tf_recorder.log[KEY_OF_TRAIN],
                    tf_recorder.log[KEY_OF_VALID],
                    tf_recorder.log[KEY_OF_TEST]
                ]
                self.count_mortality = [
                    tf_recorder.log[KEY_OF_TRAIN + KEY_OF_DEATH],
                    tf_recorder.log[KEY_OF_VALID + KEY_OF_DEATH],
                    tf_recorder.log[KEY_OF_TEST + KEY_OF_DEATH]
                ]
                self.count_alive = [
                    tf_recorder.log[KEY_OF_TRAIN + KEY_OF_ALIVE],
                    tf_recorder.log[KEY_OF_VALID + KEY_OF_ALIVE],
                    tf_recorder.log[KEY_OF_TEST + KEY_OF_ALIVE]
                ]

        data_handler = Handler(self.tf_recorder)
        self.predict(self.h, self.p, self.y_test, is_cross_valid=False)
        self.set_performance()
        self.show_performance()
        self.save_score(data_handler=data_handler,
                        best_epoch=self.best_epoch,
                        num_of_dimension=self.num_of_dimension,
                        learning_rate=self.learning_rate)
