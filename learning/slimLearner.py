from DMP.learning.neuralNet import TensorModel
from DMP.modeling.tf_recoder import to_tfrecords
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
from urllib.request import urlopen
import sys
import cv2
SLIM_PATH = '/home/nlp207/Project/models/research/slim'
sys.path.append(SLIM_PATH)

from preprocessing import vgg_preprocessing

ALIVE_DIR = 'alive'
DEATH_DIR = 'death'
BATCH_SIZE = 32
VGG_PATH = 'dataset/images/save/vgg_16.ckpt'


class SlimLearner(TensorModel):
    def __init__(self):
        super().__init__(is_cross_valid=True)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()

    def show(self):
        weights = slim.variable('weights',
                                shape=[10, 10, 3, 3],
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                regularizer=slim.l2_regularizer(0.05),
                                device='/CPU:0')

        with tf.Session() as sess:
            weights = sess.run(weights)
            print(weights)

    # 0. mnist 불러오기
    @staticmethod
    def mnist_load():
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

        # Train - Image
        train_x = train_x.astype('float32') / 255
        # Train - Label(OneHot)
        train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)

        # Test - Image
        test_x = test_x.astype('float32') / 255
        # Test - Label(OneHot)
        test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)

        return (train_x, train_y), (test_x, test_y)

    def run_fine_tuning(self, x_train, y_train):
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        print(len(x_train))
        print(len(y_train))
        # to_tfrecords(x_train, y_train, 'tf_record')

        train_filename = 'train.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)
        for i in range(len(x_train)):
            # print how many images are saved every 1000 images
            'Train data: {}/{}'.format(i, len(x_train))
            sys.stdout.flush()

            # Load the image
            img = self.load_image(x_train[i])
            label = y_train[i][0]
            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    @staticmethod
    def load_image(addr):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(addr)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img
        # for i, j in zip(x_train, y_train):
        #     print(i, j)
            # print(np.array(i).shape, np.array(j).shape)
        #
        # image_size = vgg.vgg_16.default_image_size  # image_size = 224

        # exculde = ['vgg_16/fc8']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exculde)
        #
        # saver = tf.train.Saver(variables_to_restore)
        # with tf.Session() as sess:
        #     saver.restore(sess, VGG_PATH)
        #
        #     model_variables = slim.get_model_variables()
        #
        #     print(model_variables)
        #
        # print(model_variables)
        # print(x_train.shape)
        # print(slim.get_model_variables('vgg_16'))


        # load_vars = slim.assign_from_checkpoint_fn(VGG_PATH, slim.get_model_variables('vgg_16'))
        #
        # with tf.Session() as sess:
        #     load_vars(sess)

        # # vgg_preprocessing을 이용해 전처리 수행
        # processed_img = vgg_preprocessing.preprocess_image(x_train[0],
        #                                                    image_size,
        #                                                    image_size,
        #                                                    is_training=False)
        #
        # print(processed_img)
        #
        # processed_images = tf.expand_dims(processed_img, 0)

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
