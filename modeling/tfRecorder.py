import cv2
import tensorflow as tf
import numpy as np
import json
from os import path
from .variables import EXTENSION_OF_IMAGE
from DMP.learning.variables import IMAGE_RESIZE, DO_NORMALIZE
from DMP.utils.progress_bar import show_progress_bar
from PIL import Image

EXTENSION_OF_TF_RECORD = ".tfrecords"
KEY_OF_TRAIN = "train_"
KEY_OF_TEST = "test_"


class TfRecorder:
    def __init__(self, tf_record_path):
        self.__tf_record_path = tf_record_path
        self.shape = None
        self.shape_file_name = "Shape.txt"
        self.n_fold = int()
        self.__load_shape()
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    @property
    def tf_record_path(self):
        return self.__tf_record_path

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
        self.__to_tf_records(train_image_list, train_label_list, key=KEY_OF_TRAIN + str(self.n_fold))
        self.__to_tf_records(test_image_list, test_label_list, key=KEY_OF_TEST + str(self.n_fold))
        self.__save_shape()

    def __to_tf_records(self, target_image_list, target_label_list, key):
        tf_record_path = self.tf_record_path + key + EXTENSION_OF_TF_RECORD
        total_len = len(target_image_list)

        with tf.python_io.TFRecordWriter(path=tf_record_path, options=self.options) as writer:
            for i in range(len(target_image_list)):
                img_path = target_image_list[i]
                record_name = self.__get_record_name_from_img_path(img_path)

                img = self.__load_image(img_path)
                label = target_label_list[i]
                # Create a feature

                feature = {'label': self._int64_feature(label),
                           'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'name': self._bytes_feature(record_name.encode('utf-8'))}

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

                show_progress_bar(i + 1, total_len, prefix="Save " + key.rjust(8) + EXTENSION_OF_TF_RECORD)

    def __save_shape(self):
        if self.shape:
            with open(self.tf_record_path + self.shape_file_name, 'w') as w_file:
                json.dump(list(self.shape), w_file)
                self.shape = None

    def __load_shape(self):
        tf_shape_path = self.tf_record_path + self.shape_file_name

        if path.isfile(tf_shape_path):
            with open(tf_shape_path, 'r') as r_file:
                self.shape = json.load(r_file)

    @staticmethod
    def __get_record_name_from_img_path(img_path):
        img_path = img_path.split('/')
        num_of_patient = img_path[-2]
        num_of_image = img_path[-1].split(EXTENSION_OF_IMAGE)[0]

        return num_of_patient + "_" + num_of_image + EXTENSION_OF_TF_RECORD

    def get_img_from_tf_records(self, tf_record_path):
        # record_name = self.__get_record_name_from_img_path(img_path)

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64),
                   'name': tf.FixedLenFeature([], tf.string)}
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

        if self.shape:
            image = tf.reshape(image, self.shape)
        else:
            image = tf.reshape(image, [IMAGE_RESIZE, IMAGE_RESIZE, 3])

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int8)
        name = tf.cast(features['name'], tf.string)

        print("\nSuccess to read -", tf_record_path, "\n")

        if DO_NORMALIZE:
            return image / 255, label, name
        else:
            return tf.cast(image, tf.uint8), label, name

    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        if not self.shape:
            self.shape = img.shape

        return img
