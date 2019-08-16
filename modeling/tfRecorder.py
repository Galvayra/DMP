import cv2
import tensorflow as tf
import numpy as np
import json
from os import path
from .variables import EXTENSION_OF_IMAGE
from DMP.learning.variables import IMAGE_RESIZE, DO_NORMALIZE
from DMP.utils.progress_bar import show_progress_bar

EXTENSION_OF_TF_RECORD = ".tfrecords"
KEY_OF_SHAPE = "shape"
KEY_OF_TRAIN = "train_"
KEY_OF_TEST = "test_"
KEY_OF_DIM = "dim_"


class TfRecorder:
    def __init__(self, tf_record_path):
        self.__tf_record_path = tf_record_path
        self.log = dict()
        self.log_file_name = "Log.txt"
        self.n_fold = int()
        self.__load_log()
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    @property
    def tf_record_path(self):
        return self.__tf_record_path

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def to_tf_records(self, train_image_list, train_label_list, test_image_list, test_label_list):
        self.n_fold += 1

        # set log for training
        if KEY_OF_SHAPE not in self.log:
            self.log[KEY_OF_SHAPE] = self.__get_shape(train_image_list[0][1])
        self.log[KEY_OF_DIM + str(self.n_fold)] = len(train_image_list[0][0])
        self.log[KEY_OF_TRAIN + str(self.n_fold)] = len(train_label_list)
        self.log[KEY_OF_TEST + str(self.n_fold)] = len(test_label_list)

        self.__to_tf_records(train_image_list, train_label_list, key=KEY_OF_TRAIN + str(self.n_fold))
        self.__to_tf_records(test_image_list, test_label_list, key=KEY_OF_TEST + str(self.n_fold))

    def __to_tf_records(self, target_image_list, target_label_list, key):
        tf_record_path = self.tf_record_path + key + EXTENSION_OF_TF_RECORD
        total_len = len(target_image_list)

        with tf.python_io.TFRecordWriter(path=tf_record_path, options=self.options) as writer:
            for i in range(len(target_image_list)):
                vector = target_image_list[i][0]
                img_path = target_image_list[i][1]
                record_name = self.__get_record_name_from_img_path(img_path)

                img = self.__load_image(img_path)
                label = target_label_list[i]
                # Create a feature

                feature = {'label': self._int64_feature(label),
                           'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'name': self._bytes_feature(record_name.encode('utf-8')),
                           'vector': self._float_feature(vector)}

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

                show_progress_bar(i + 1, total_len, prefix="Save " + key.rjust(8) + EXTENSION_OF_TF_RECORD)

    def save(self):
        with open(self.tf_record_path + self.log_file_name, 'w') as w_file:
            json.dump(self.log, w_file, indent=4)

    def __load_log(self):
        tf_log_path = self.tf_record_path + self.log_file_name

        if path.isfile(tf_log_path):
            with open(tf_log_path, 'r') as r_file:
                self.log = json.load(r_file)

    @staticmethod
    def __get_record_name_from_img_path(img_path):
        img_path = img_path.split('/')
        num_of_patient = img_path[-2]
        num_of_image = img_path[-1].split(EXTENSION_OF_IMAGE)[0]

        return num_of_patient + "_" + num_of_image + EXTENSION_OF_TF_RECORD

    def get_img_from_tf_records(self, tf_record_path):
        # record_name = self.__get_record_name_from_img_path(img_path)

        # feature = {'image': tf.FixedLenFeature([], tf.string),
        #            'label': tf.FixedLenFeature([], tf.int64),
        #            'name': tf.FixedLenFeature([], tf.string),
        #            'vector': tf.FixedLenFeature([], tf.float32)}

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64),
                   'name': tf.FixedLenFeature([], tf.string),
                   'vector': tf.VarLenFeature(tf.float32)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tf_record_path], num_epochs=1)
        # print(filename_queue)
        # Define a reader and read the next record

        reader = tf.TFRecordReader(options=self.options)
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, self.log[KEY_OF_SHAPE])

        # because a number of dim is same whenever fold
        # vector = tf.reshape(features['vector'], [self.log[KEY_OF_DIM + "1"]])
        vector = tf.sparse_tensor_to_dense(features['vector'], default_value=0)
        vector = tf.reshape(vector, [self.log[KEY_OF_DIM + "1"]])

        # Cast label data into
        label = tf.reshape(features['label'], [1])
        label = tf.cast(label, tf.int8)
        name = tf.cast(features['name'], tf.string)

        print("\nSuccess to read -", tf_record_path, "\n")

        if DO_NORMALIZE:
            return vector, image / 255, label, name
        else:
            return vector, tf.cast(image, tf.uint8), label, name

    @staticmethod
    def __load_image(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        return img

    @staticmethod
    def __get_shape(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)

        return img.shape
