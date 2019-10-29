import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import *
from DMP.utils.arg_encoding import VERSION, LOG_NAME, NUM_OF_IMPORTANT, DO_CROSS_ENTROPY, DO_ENCODE_IMAGE, \
    IS_CROSS_VALID
from os import path
from DMP.dataset.images.variables import CT_IMAGE_PATH, CT_IMAGE_ALL_PATH, IMAGE_PATH
from DMP.dataset.variables import DATA_PATH
from DMP.modeling.tfRecorder import TfRecorder
from DMP.modeling.imageMaker import ImageMaker
from sklearn.model_selection import KFold
from DMP.learning.variables import NUM_OF_K_FOLD
from DMP.utils.progress_bar import show_progress_bar
import json
import os
import shutil
import random

SEED = 1
DO_SHUFFLE = True
TRAIN_RATIO = 8
VALID_RATIO = TRAIN_RATIO + (10 - TRAIN_RATIO) / 2

TRAIN_RATIO /= 10
VALID_RATIO /= 10


###
# list of KEY
#   total == train set + validation set + test set
#   train == train set
#   valid == validation set
#   test == test set
###
class VectorMaker:
    # must using DataParser or DataHandler
    def __init__(self, data_handler):
        # dataHandler_dict = { KEY : handler }
        self.dataHandler_dict = {key: handler for key, handler in data_handler.items()}
        self.__y_data = self.dataHandler_dict[KEY_TOTAL].y_data
        self.__len_data = len(self.dataHandler_dict[KEY_TOTAL].y_data)

        # {
        #   feature: { 0: ["D", "header_name"], ... , n(dimensionality): ["CZ", "header_name"] }
        #   x_train: [ vector 1, ... vector n ], ... x_test, x_valid , ... , y_valid
        # }
        self.__vector_matrix = OrderedDict()
        self.__vector_matrix = {
            "feature": dict(),
            "x_train": dict(),
            "y_train": list(),
            "x_valid": dict(),
            "y_valid": list(),
            "x_test": dict(),
            "y_test": list()
        }

        self.__dump_path = path.dirname(path.abspath(__file__)) + "/" + DUMP_PATH
        self.__image_path = path.dirname(path.dirname(path.abspath(__file__))) + "/" + DATA_PATH + IMAGE_PATH
        self.__tf_record_path = path.dirname(path.abspath(__file__)) + "/" + TF_RECORD_PATH + op.FILE_VECTOR + "/"

        self.__pickles_path = path.dirname(path.abspath(__file__)) + "/" + IMAGE_PICKLES_PATH + op.FILE_VECTOR + "/"

        # initialize TfRecorder class
        self.tf_recorder = TfRecorder(self.tf_record_path,
                                      do_encode_image=DO_ENCODE_IMAGE,
                                      is_cross_valid=IS_CROSS_VALID)

        # initialize ImageMaker class
        self.image_maker = ImageMaker(self.pickles_path)

    @property
    def y_data(self):
        return self.__y_data

    @property
    def len_data(self):
        return self.__len_data

    @property
    def vector_matrix(self):
        return self.__vector_matrix

    @property
    def dump_path(self):
        return self.__dump_path

    @property
    def image_path(self):
        return self.__image_path

    @property
    def tf_record_path(self):
        return self.__tf_record_path

    @property
    def pickles_path(self):
        return self.__pickles_path

    def encoding(self):
        ct_image_path = self.image_path + CT_IMAGE_PATH + CT_IMAGE_ALL_PATH

        # init encoder and fit it
        encoder = MyOneHotEncoder(ver=VERSION, log_name=LOG_NAME, num_of_important=NUM_OF_IMPORTANT,
                                  ct_image_path=ct_image_path)
        encoder.fit(self.dataHandler_dict[KEY_TOTAL])

        # initialize dictionary of matrix after encoding
        matrix_dict = {
            ("x_train", "y_train", KEY_TRAIN): encoder.transform2matrix(self.dataHandler_dict[KEY_TRAIN]),
            ("x_valid", "y_valid", KEY_VALID): encoder.transform2matrix(self.dataHandler_dict[KEY_VALID]),
            ("x_test", "y_test", KEY_TEST): encoder.transform2matrix(self.dataHandler_dict[KEY_TEST])
        }

        self.__set_vector_matrix_feature(encoder.get_feature_dict())
        self.__set_vector_matrix(matrix_dict)

        if DO_ENCODE_IMAGE:
            image_matrix_dict = {
                KEY_IMG_TRAIN: encoder.transform2image_matrix(self.dataHandler_dict[KEY_TRAIN]),
                KEY_IMG_VALID: encoder.transform2image_matrix(self.dataHandler_dict[KEY_VALID]),
                KEY_IMG_TEST: encoder.transform2image_matrix(self.dataHandler_dict[KEY_TEST])
            }

            self.__set_vector_matrix4image(image_matrix_dict)

        del self.dataHandler_dict

    def __set_vector_matrix_feature(self, feature_dict):
        self.vector_matrix["feature"] = feature_dict
        print("# of dim -", len(feature_dict), "\n\n")

    def __set_vector_matrix(self, matrix_dict):
        def __copy(x_target, y_target, y_data):
            for index in range(len(matrix[KEY_NAME_OF_MERGE_VECTOR])):
                for class_of_column in x_target:
                    x_target[class_of_column].append(matrix[class_of_column][index])

                if DO_CROSS_ENTROPY:
                    if y_data[index] == [1]:
                        y = [0, 1]
                    else:
                        y = [1, 0]
                else:
                    y = y_data[index]

                y_target.append(y)

        # initialize x data in self.vector_matrix
        for key, matrix in matrix_dict.items():
            self.vector_matrix[key[0]] = {class_of_column: list() for class_of_column in matrix}

        # copy x data in self.vector_matrix
        for key, matrix in matrix_dict.items():
            __copy(self.vector_matrix[key[0]], self.vector_matrix[key[1]], self.dataHandler_dict[key[2]].y_data)

    def __set_vector_matrix4image(self, matrix_dict):
        for key, matrix in matrix_dict.items():
            self.vector_matrix[key] = matrix

    @staticmethod
    def __mkdir_records(_path):
        if os.path.isdir(_path):
            print("\nThe directory for record is already existed -", _path, "\n")
            while True:
                do_continue = input("Do you want to re-encoding? (y/n) - ").lower()
                if do_continue == 'n':
                    return False
                elif do_continue == 'y':
                    shutil.rmtree(_path)
                    os.mkdir(_path)
                    break
        
        return True

    def build_tf_records(self):
        if VERSION == 1 and self.__mkdir_records(self.tf_record_path):
            x_train, y_train = self.__get_set(key="train")
            x_valid, y_valid = self.__get_set(key="valid")
            x_test, y_test = self.__get_set(key="test")

            # erase shuffle set
            x_data, y_data = self.__get_shuffle_set(x_train + x_valid + x_test, y_train + y_valid + y_test)

            # shuffle data for avoiding over-fitting
            if IS_CROSS_VALID:
                print("This scope will be implemented")
                # TODO implement k-fold cross validation
                exit(-1)
                for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                    self.tf_recorder.to_tf_records(x_train, y_train, key="train")
                    self.tf_recorder.to_tf_records(x_test, y_test, key="test")
            else:
                # if DO_ENCODE_IMAGE:
                #     i_train, i_valid = int(len(y_data) * TRAIN_RATIO), int(len(y_data) * VALID_RATIO)
                #     x_train, x_valid, x_test = x_data[:i_train], x_data[i_train:i_valid], x_data[i_valid:]
                #     y_train, y_valid, y_test = y_data[:i_train], y_data[i_train:i_valid], y_data[i_valid:]

                self.tf_recorder.to_tf_records(x_train, y_train, key="train")
                self.tf_recorder.to_tf_records(x_valid, y_valid, key="valid")
                self.tf_recorder.to_tf_records(x_test, y_test, key="test")

            self.tf_recorder.save()
            print("success build tf records! (in the -", self.tf_record_path + ")\n\n")

    def __add_key_value_in_dict(self, key, value):
        if KEY_TF_NAME not in self.vector_matrix:
            self.vector_matrix[KEY_TF_NAME] = dict()

        self.vector_matrix[KEY_TF_NAME][key] = value

    def __get_set(self, key, converter="image_maker"):
        """

        :param key:
        :return:
        x_data = [
                    [ vector_1, img_path_1], [ vector_2, img_path_2], ... , [ vector_N, img_path_N]
                ]

        y_data = [ label_1, label_2, ... , label_N ]
        """
        _x_data, _y_data = list(), list()

        x_target = self.vector_matrix["x_" + key][KEY_NAME_OF_MERGE_VECTOR]
        y_target = self.vector_matrix["y_" + key]

        if key == "train":
            img_key = KEY_IMG_TRAIN
        elif key == "valid":
            img_key = KEY_IMG_VALID
        else:
            img_key = KEY_IMG_TEST

        if converter == "image_maker":
            converter = self.image_maker
        else:
            converter = self.tf_recorder

        if img_key in self.vector_matrix:
            x_path_target = self.vector_matrix[img_key]

            for vector, img_paths, y in zip(x_target, x_path_target, y_target):
                for img_path in img_paths:
                    _y_data.append(y)
                    _x_data.append([vector, img_path])
                    tf_name = converter.get_img_name_from_path(img_path)
                    self.__add_key_value_in_dict(tf_name, [vector, y])
        else:
            for vector, y in zip(x_target, y_target):
                _y_data.append(y)
                tf_name = key + '_' + str(len(_y_data))
                _x_data.append([vector, tf_name])
                self.__add_key_value_in_dict(tf_name, [vector, y])

        return _x_data, _y_data

    @staticmethod
    def __get_shuffle_set(x_data, y_data):
        if DO_SHUFFLE:
            print("\nApply shuffle to the dataset for avoiding over-fitting\n\n")
            random.seed(SEED)
            c = list(zip(x_data, y_data))
            random.shuffle(c)
            x_data, y_data = zip(*c)

        return x_data, y_data

    @staticmethod
    def __get_data_matrix(_data, _index_list):
        return [_data[i] for i in _index_list]

    def __data_generator(self, x_data, y_data):
        cv = KFold(n_splits=NUM_OF_K_FOLD, random_state=1, shuffle=True)

        for train_index_list, test_index_list in cv.split(x_data, y_data):
            x_train = self.__get_data_matrix(x_data, train_index_list)
            y_train = self.__get_data_matrix(y_data, train_index_list)
            x_test = self.__get_data_matrix(x_data, test_index_list)
            y_test = self.__get_data_matrix(y_data, test_index_list)

            yield x_train, y_train, x_test, y_test

    def build_pillow_img(self):
        if VERSION == 1 and DO_ENCODE_IMAGE == 1 and self.__mkdir_records(self.pickles_path):
            x_train, y_train = self.__get_set(key="train")
            x_valid, y_valid = self.__get_set(key="valid")
            x_test, y_test = self.__get_set(key="test")

            x_data, y_data = x_train + x_valid + x_test, y_train + y_valid + y_test

            total_len = len(x_data)

            for i, x in enumerate(x_data):
                img_path = x[1]
                self.image_maker.image2vector(img_path)
                show_progress_bar(i + 1, total_len, prefix="Save pickles of image")

    def dump(self):
        if op.FILE_VECTOR:
            file_name = self.dump_path + op.FILE_VECTOR
        else:
            file_name = self.dump_path + DUMP_FILE

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_matrix, outfile, indent=4)
            print("\n=========================================================\n")
            print("success make dump file! - file name is", file_name, "\n\n\n")

    def show_vector_info(self):
        def __counting_mortality(_data):
            count = 0

            if DO_CROSS_ENTROPY:
                death_vector = [0, 1]
            else:
                death_vector = [1]

            for _d in _data:
                if _d == death_vector:
                    count += 1

            return count

        if DO_SHOW_VECTOR_INFO:
            len_x_train = len(self.vector_matrix["x_train"]["merge"])
            len_x_valid = len(self.vector_matrix["x_valid"]["merge"])
            len_x_test = len(self.vector_matrix["x_test"]["merge"])

            len_y_train = __counting_mortality(self.vector_matrix["y_train"])
            len_y_valid = __counting_mortality(self.vector_matrix["y_valid"])
            len_y_test = __counting_mortality(self.vector_matrix["y_test"])

            print("\nAll   total count -", str(len_x_train + len_x_valid + len_x_test).rjust(4),
                  "\tDeath count -", str(len_y_train + len_y_valid + len_y_test).rjust(4))
            print("Train total count -", str(len_x_train).rjust(4),
                  "\tDeath count -", str(len_y_train).rjust(4))
            print("Valid total count -", str(len_x_valid).rjust(4),
                  "\tDeath count -", str(len_y_valid).rjust(4))
            print("Test  total count -", str(len_x_test).rjust(4),
                  "\tDeath count -", str(len_y_test).rjust(4), "\n\n")
