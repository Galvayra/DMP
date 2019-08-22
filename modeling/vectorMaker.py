import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH, KEY_TOTAL, KEY_TRAIN, KEY_VALID, KEY_TEST, KEY_NAME_OF_MERGE_VECTOR, \
    KEY_IMG_TEST, KEY_IMG_TRAIN, KEY_IMG_VALID, TF_RECORD_PATH
from DMP.utils.arg_encoding import VERSION, LOG_NAME, NUM_OF_IMPORTANT, DO_CROSS_ENTROPY
from os import path
from DMP.dataset.images.variables import CT_IMAGE_PATH, CT_IMAGE_ALL_PATH, IMAGE_PATH
from DMP.dataset.variables import DATA_PATH
from DMP.modeling.tfRecorder import TfRecorder
from sklearn.model_selection import KFold
from DMP.learning.variables import NUM_OF_K_FOLD
import json
import os
import shutil


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

    def encoding(self, encode_image=False):
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

        if encode_image:
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

    def build_tf_records(self):
        if os.path.isdir(self.tf_record_path):
            print("\nThe directory for tfrecord is already existed -", self.tf_record_path, "   \n\n")
        else:
            os.mkdir(self.tf_record_path)

            x_train, y_train = self.__get_set(key="train")
            x_valid, y_valid = self.__get_set(key="valid")
            x_test, y_test = self.__get_set(key="test")
            x_data, y_data = x_train + x_valid + x_test, y_train + y_valid + y_test

            tf_recorder = TfRecorder(self.tf_record_path)
            for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                tf_recorder.to_tf_records(x_train, y_train, x_test, y_test)

            tf_recorder.save()
            print("\n=========================================================\n")
            print("success build tf records! (in the -", self.tf_record_path + ")\n\n\n")

    def __get_set(self, key):
        _x_data, _y_data = list(), list()

        x_target = self.vector_matrix["x_" + key][KEY_NAME_OF_MERGE_VECTOR]
        y_target = self.vector_matrix["y_" + key]

        if key == "train":
            img_key = KEY_IMG_TRAIN
        elif key == "valid":
            img_key = KEY_IMG_VALID
        else:
            img_key = KEY_IMG_TEST

        if img_key in self.vector_matrix:
            x_path_target = self.vector_matrix[img_key]

            for vector, img_paths, y in zip(x_target, x_path_target, y_target):
                for img_path in img_paths:
                    _x_data.append([vector, img_path])
                    _y_data.append(y)
        else:
            for vector, y in zip(x_target, y_target):
                _x_data.append([vector])
                _y_data.append(y)

        return _x_data, _y_data

    @staticmethod
    def __get_data_matrix(_data, _index_list):
        return [_data[i] for i in _index_list]

    def __data_generator(self, x_data, y_data):
        cv = KFold(n_splits=NUM_OF_K_FOLD, random_state=0, shuffle=False)

        for train_index_list, test_index_list in cv.split(x_data, y_data):
            x_train = self.__get_data_matrix(x_data, train_index_list)
            y_train = self.__get_data_matrix(y_data, train_index_list)
            x_test = self.__get_data_matrix(x_data, test_index_list)
            y_test = self.__get_data_matrix(y_data, test_index_list)

            yield x_train, y_train, x_test, y_test

    def dump(self, do_show=True):
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

        if op.FILE_VECTOR:
            file_name = self.dump_path + op.FILE_VECTOR
        else:
            file_name = self.dump_path + DUMP_FILE

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_matrix, outfile, indent=4)
            print("\n=========================================================\n\n")
            print("success make dump file! - file name is", file_name)

        if do_show:
            len_x_train = len(self.vector_matrix["x_train"]["merge"])
            len_x_valid = len(self.vector_matrix["x_valid"]["merge"])
            len_x_test = len(self.vector_matrix["x_test"]["merge"])

            len_y_train = __counting_mortality(self.vector_matrix["y_train"])
            len_y_valid = __counting_mortality(self.vector_matrix["y_valid"])
            len_y_test = __counting_mortality(self.vector_matrix["y_test"])

            print("\nAll   total count -", str(len_x_train + len_x_valid + len_x_test).rjust(4),
                  "\tmortality count -", str(len_y_train + len_y_valid + len_y_test).rjust(4))
            print("Train total count -", str(len_x_train).rjust(4),
                  "\tmortality count -", str(len_y_train).rjust(4))
            print("Valid total count -", str(len_x_valid).rjust(4),
                  "\tmortality count -", str(len_y_valid).rjust(4))
            print("Test  total count -", str(len_x_test).rjust(4),
                  "\tmortality count -", str(len_y_test).rjust(4), "\n\n")
