import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH, KEY_TOTAL, KEY_TRAIN, KEY_VALID, KEY_TEST, KEY_NAME_OF_MERGE_VECTOR, \
    KEY_IMG_TEST, KEY_IMG_TRAIN, KEY_IMG_VALID
from DMP.utils.arg_encoding import VERSION, LOG_NAME, NUM_OF_IMPORTANT, DO_CROSS_ENTROPY
from os import path
from DMP.dataset.images.variables import CT_IMAGE_PATH, CT_IMAGE_ALL_PATH, IMAGE_PATH
from DMP.dataset.variables import DATA_PATH
import json


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
            # ct_dict = self.__load_ct_dict()

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

    # def __load_ct_dict(self):
    #     with open(self.image_path + IMAGE_LOG_PATH + IMAGE_LOG_NAME, 'r') as r_file:
    #         return json.load(r_file)

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
