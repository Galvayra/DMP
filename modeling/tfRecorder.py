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
KEY_OF_TRAIN = "train"
KEY_OF_VALID = "valid"
KEY_OF_TEST = "test"
KEY_OF_DIM = "dim"
KEY_OF_DIM_OUTPUT = "dim_output"
KEY_OF_ALIVE = "alive"
KEY_OF_DEATH = "death"
KEY_OF_IS_CROSS_VALID = "is_cross_valid"


class TfRecorder:
    def __init__(self, tf_record_path, do_encode_image=False, is_cross_valid=False):
        self.__tf_record_path = tf_record_path
        self.log = dict()
        self.log_file_name = "Log.txt"
        # self.n_fold = int()
        self.__do_encode_image = do_encode_image
        self.__load_log()
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.__is_cross_valid = is_cross_valid

    @property
    def tf_record_path(self):
        return self.__tf_record_path

    @property
    def do_encode_image(self):
        return self.__do_encode_image

    @do_encode_image.setter
    def do_encode_image(self, do_encode_image):
        self.__do_encode_image = do_encode_image

    @property
    def is_cross_valid(self):
        return self.__is_cross_valid

    @is_cross_valid.setter
    def is_cross_valid(self, is_cross_valid):
        self.__is_cross_valid = is_cross_valid

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

    def to_tf_records(self, x_data, y_data, key=KEY_OF_TRAIN):
        # set log for training
        if self.do_encode_image and KEY_OF_SHAPE not in self.log:
            self.log[KEY_OF_SHAPE] = self.__get_shape(x_data[0][1])

        if KEY_OF_IS_CROSS_VALID not in self.log:
            self.log[KEY_OF_IS_CROSS_VALID] = self.is_cross_valid

        self.log[key + KEY_OF_DIM] = len(x_data[0][0])
        self.log[key + KEY_OF_DIM_OUTPUT] = len(y_data[0])
        self.log[key] = len(y_data)

        death_count = int()
        for y in y_data:
            if y == [1]:
                death_count += 1

        self.log[key + KEY_OF_ALIVE] = self.log[key] - death_count
        self.log[key + KEY_OF_DEATH] = death_count
        self.__to_tf_records(x_data, y_data, key=key)

    def __to_tf_records(self, target_image_list, target_label_list, key):
        tf_record_path = self.tf_record_path + key + EXTENSION_OF_TF_RECORD
        total_len = len(target_image_list)

        with tf.python_io.TFRecordWriter(path=tf_record_path, options=self.options) as writer:
            """
            feature {
                'vector': vector of numeric and class from medical data
                'label': label
                'image': vector of ct image
                'name': file name of ct image
            }
            """
            for i in range(len(target_image_list)):
                vector = target_image_list[i][0]
                label = target_label_list[i]

                if self.do_encode_image:
                    img_path = target_image_list[i][1]
                    record_name = self.__get_record_name_from_img_path(img_path)
                    img = self.__load_image(img_path)

                    feature = {
                        'vector': self._float_feature(vector),
                        'label': self._int64_feature(label),
                        'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                        'name': self._bytes_feature(record_name.encode('utf-8'))
                    }
                else:
                    feature = {
                        'vector': self._float_feature(vector),
                        'label': self._int64_feature(label),
                        'image': self._int64_feature(0),
                        'name': self._int64_feature(0)
                    }

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                show_progress_bar(i + 1, total_len, prefix="Save " + key.rjust(8) + EXTENSION_OF_TF_RECORD)

    def save(self):
        print("\n\n\n======== Set of TfRecord ========\n")
        print("# of dimensions  -", str(self.log[KEY_OF_TRAIN + KEY_OF_DIM]).rjust(4))
        print("Training   Count -", str(self.log[KEY_OF_TRAIN]).rjust(4),
              "\t Alive Count -", str(self.log[KEY_OF_TRAIN + KEY_OF_ALIVE]).rjust(4),
              "\t Death Count -", str(self.log[KEY_OF_TRAIN + KEY_OF_DEATH]).rjust(3))
        if self.is_cross_valid:
            print("Validation Count -", str(self.log[KEY_OF_VALID]).rjust(4),
                  "\t Alive Count -", str(self.log[KEY_OF_VALID + KEY_OF_ALIVE]).rjust(4),
                  "\t Death Count -", str(self.log[KEY_OF_VALID + KEY_OF_DEATH]).rjust(3))
        print("Test       Count -", str(self.log[KEY_OF_TEST]).rjust(4),
              "\t Alive Count -", str(self.log[KEY_OF_TEST + KEY_OF_ALIVE]).rjust(4),
              "\t Death Count -", str(self.log[KEY_OF_TEST + KEY_OF_DEATH]).rjust(3), '\n')

        with open(self.tf_record_path + self.log_file_name, 'w') as w_file:
            json.dump(self.log, w_file, indent=4)

    def __load_log(self):
        tf_log_path = self.tf_record_path + self.log_file_name

        if path.isfile(tf_log_path):
            with open(tf_log_path, 'r') as r_file:
                self.log = json.load(r_file)

                if KEY_OF_SHAPE not in self.log:
                    self.do_encode_image = False
                else:
                    self.do_encode_image = True

                self.is_cross_valid = self.log[KEY_OF_IS_CROSS_VALID]

    @staticmethod
    def __get_record_name_from_img_path(img_path):
        img_path = img_path.split('/')
        num_of_patient = img_path[-2]
        num_of_image = img_path[-1].split(EXTENSION_OF_IMAGE)[0]

        return num_of_patient + "_" + num_of_image + EXTENSION_OF_TF_RECORD

    def _parse_func(self, serialized_example):
        if self.do_encode_image:
            feature = {
                'vector': tf.VarLenFeature(tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'name': tf.FixedLenFeature([], tf.string)
            }
        else:
            feature = {
                'vector': tf.VarLenFeature(tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.int64),
                'name': tf.FixedLenFeature([], tf.int64)
            }

        features = tf.parse_single_example(serialized_example, features=feature)

        # because a number of dim is same whenever fold
        # vector = tf.reshape(features['vector'], [self.log[KEY_OF_DIM + "1"]])
        vector = tf.sparse_tensor_to_dense(features['vector'], default_value=0)
        vector = tf.reshape(vector, [self.log[KEY_OF_TRAIN + KEY_OF_DIM]])

        # Cast label data into
        label = tf.reshape(features['label'], [1])
        label = tf.cast(label, tf.int8)

        if self.do_encode_image:
            image = tf.decode_raw(features['image'], tf.float32)
            image = tf.reshape(image, self.log[KEY_OF_SHAPE])

            name = tf.cast(features['name'], tf.string)

            if DO_NORMALIZE:
                image /= image
            else:
                image = tf.cast(image, tf.uint8)
        else:
            image = tf.reshape(features['image'], [1])
            image = tf.cast(image, tf.int8)

            name = tf.reshape(features['name'], [1])
            name = tf.cast(name, tf.int8)

        return vector, label, image, name

        # return tf.parse_single_example(example_proto, feature)

    def get_img_from_tf_records(self, tf_record_path):
        raw_image_dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="GZIP")
        return raw_image_dataset.map(self._parse_func)

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
