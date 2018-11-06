import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH, KEY_TOTAL, KEY_TRAIN, KEY_VALID, KEY_TEST
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
    def __init__(self, **data_handler):
        # dataHandler_dict = { KEY : handler }
        self.dataHandler_dict = {key: handler for key, handler in data_handler.items()}
        self.__y_data = self.dataHandler_dict[KEY_TOTAL].y_data
        self.__len_data = len(self.dataHandler_dict[KEY_TOTAL].y_data)

        # { x_train: [ vector 1, ... vector n ], ... x_test, x_valid , ... , y_valid }
        self.__vector_matrix = OrderedDict()
        self.__vector_matrix = {
            "x_train": dict(),
            "y_train": list(),
            "x_valid": dict(),
            "y_valid": list(),
            "x_test": dict(),
            "y_test": list()
        }

    @property
    def y_data(self):
        return self.__y_data

    @property
    def len_data(self):
        return self.__len_data

    @property
    def vector_matrix(self):
        return self.__vector_matrix

    def encoding(self):
        # init encoder and fit it
        encoder = MyOneHotEncoder(self.dataHandler_dict[KEY_TOTAL])
        encoder.encoding()

        matrix_train = encoder.fit(self.dataHandler_dict[KEY_TRAIN])
        matrix_valid = encoder.fit(self.dataHandler_dict[KEY_VALID])
        matrix_test = encoder.fit(self.dataHandler_dict[KEY_TEST])
        # encoder.show_vectors(*self.dataHandler.header_list)

        print(len(matrix_train), len(matrix_valid), len(matrix_test))
        exit(-1)
        self.__set_vector_matrix(encoder.vector_matrix)

        del self.dataHandler_dict

    def __set_vector_matrix(self, vector_matrix):
        def __copy(x_target, y_target, y_data):
            for _class_of_column in x_target:
                x_target[_class_of_column].append(vector_matrix[_class_of_column][index])

            y_target.append(y_data)

        self.vector_matrix["x_train"] = {class_of_column: list() for class_of_column in vector_matrix}
        self.vector_matrix["x_valid"] = {class_of_column: list() for class_of_column in vector_matrix}
        self.vector_matrix["x_test"] = {class_of_column: list() for class_of_column in vector_matrix}

        for index in range(self.len_data):
            which = self.index_dict[index]

            if which is "train":
                __copy(self.vector_matrix["x_train"], self.vector_matrix["y_train"], self.y_data[index])
            elif which is "valid":
                __copy(self.vector_matrix["x_valid"], self.vector_matrix["y_valid"], self.y_data[index])
            elif which is "test":
                __copy(self.vector_matrix["x_test"], self.vector_matrix["y_test"], self.y_data[index])

    def dump(self, do_show=True):
        def __counting_mortality(_data):
            count = 0
            for _d in _data:
                if _d == [1]:
                    count += 1

            return count

        if op.FILE_VECTOR:
            file_name = DUMP_PATH + op.FILE_VECTOR
        else:
            file_name = DUMP_PATH + DUMP_FILE

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_matrix, outfile, indent=4)
            print("success make dump file! - file name is", file_name)

        if do_show:
            print("\n\nTrain total count -", str(len(self.vector_matrix["x_train"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_train"])).rjust(4))
            print("Valid total count -", str(len(self.vector_matrix["x_valid"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_valid"])).rjust(4))
            print("Test  total count -", str(len(self.vector_matrix["x_test"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_test"])).rjust(4), "\n\n")
