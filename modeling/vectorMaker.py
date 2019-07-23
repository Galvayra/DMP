import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH, KEY_TOTAL, KEY_TRAIN, KEY_VALID, KEY_TEST, KEY_NAME_OF_MERGE_VECTOR
from DMP.utils.arg_encoding import VERSION, LOG_NAME, NUM_OF_IMPORTANT
from os import path
from DMP.dataset.images.variables import CT_IMAGE_PATH
from DMP.dataset.variables import DATA_PATH, IMAGE_PATH, IMAGE_LOG_NAME, IMAGE_LOG_PATH
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
    def __init__(self, data_handler, ct_image_path=""):
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

        self.__ct_image_path = ct_image_path
        self.__dump_path = path.dirname(path.abspath(__file__)) + "/"
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
    def ct_image_path(self):
        return self.__ct_image_path

    @property
    def dump_path(self):
        return self.__dump_path

    @property
    def image_path(self):
        return self.__image_path

    def encoding(self):
        # init encoder and fit it
        encoder = MyOneHotEncoder(ver=VERSION, log_name=LOG_NAME, num_of_important=NUM_OF_IMPORTANT)
        encoder.fit(self.dataHandler_dict[KEY_TOTAL])

        # initialize dictionary of matrix after encoding
        matrix_dict = {
            ("x_train", "y_train", KEY_TRAIN): encoder.transform2matrix(self.dataHandler_dict[KEY_TRAIN]),
            ("x_valid", "y_valid", KEY_VALID): encoder.transform2matrix(self.dataHandler_dict[KEY_VALID]),
            ("x_test", "y_test", KEY_TEST): encoder.transform2matrix(self.dataHandler_dict[KEY_TEST])
        }

        self.__set_vector_matrix_feature(encoder.get_feature_dict())
        self.__set_vector_matrix(matrix_dict)
        #
        # del self.dataHandler_dict

    def __set_vector_matrix_feature(self, feature_dict):
        self.vector_matrix["feature"] = feature_dict
        print("# of dim -", len(feature_dict), "\n\n")

    def __set_vector_matrix(self, matrix_dict):
        def __copy(x_target, y_target, y_data):
            for index in range(len(matrix[KEY_NAME_OF_MERGE_VECTOR])):
                for class_of_column in x_target:
                    x_target[class_of_column].append(matrix[class_of_column][index])

                y_target.append(y_data[index])

        # initialize x data in self.vector_matrix
        for key, matrix in matrix_dict.items():
            self.vector_matrix[key[0]] = {class_of_column: list() for class_of_column in matrix}

        # copy x data in self.vector_matrix
        for key, matrix in matrix_dict.items():
            __copy(self.vector_matrix[key[0]], self.vector_matrix[key[1]], self.dataHandler_dict[key[2]].y_data)

    def encoding_images(self):
        print(self.vector_matrix.keys())
        self.__load_ct_dict()
        exit(-1)

    def __load_ct_dict(self):
        with open(self.image_path + IMAGE_LOG_PATH + IMAGE_LOG_NAME, 'r') as r_file:
            ct_dict = json.load(r_file)
            print(ct_dict)

    def dump(self, do_show=True):
        def __counting_mortality(_data):
            count = 0
            for _d in _data:
                if _d == [1]:
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
            print("\nTrain total count -", str(len(self.vector_matrix["x_train"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_train"])).rjust(4))
            print("Valid total count -", str(len(self.vector_matrix["x_valid"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_valid"])).rjust(4))
            print("Test  total count -", str(len(self.vector_matrix["x_test"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_test"])).rjust(4), "\n\n")
