import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH
import json
import copy
import random


class VectorMaker:
    # must using DataParser or DataHandler
    def __init__(self, data_handler):
        self.dataHandler = data_handler
        self.__y_data = self.dataHandler.y_data
        self.__len_data = len(self.dataHandler.y_data)

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

        # { 0: "train", 1: "test" , ..., n: "valid" }
        self.__index_dict = self.__init_index_dict()
        self.__file_name = self.dataHandler.file_name.split('.')[0]

    @property
    def file_name(self):
        return self.__file_name

    @property
    def y_data(self):
        return self.__y_data

    @property
    def len_data(self):
        return self.__len_data

    @property
    def index_dict(self):
        return self.__index_dict

    @property
    def vector_matrix(self):
        return self.__vector_matrix

    # train:test:valid  --> 5 : 2.5 : 2.5
    # if ratio == 0.8   --> 8 :   1 :   1
    def __init_index_dict(self):
        index_dict = dict()
        data_dict = {
            "train": list(),
            "test": list(),
            "valid": list()
        }

        def __is_choice(ratio=0.5):
            if random.randrange(10) < (ratio * 10):
                return True
            else:
                return False

        for index in range(self.len_data):
            if __is_choice(op.RATIO):
                index_dict[index] = "train"
                data_dict["train"].append(index)
            elif __is_choice():
                index_dict[index] = "test"
                data_dict["test"].append(index)
            else:
                index_dict[index] = "valid"
                data_dict["valid"].append(index)
        #
        # for k in data_dict:
        #     print(k.rjust(8), "count -", len(data_dict[k]))

        return index_dict

    def encoding(self):
        # init encoder and fit it
        encoder = MyOneHotEncoder(self.dataHandler, w2v=op.USE_W2V)
        encoder.encoding()
        encoder.fit()
        # encoder.show_vectors(*self.dataHandler.header_list)

        self.__set_vector_matrix(encoder.vector_matrix)

        del self.dataHandler

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
            if op.USE_W2V:
                append_name = "_w2v_"
            else:
                append_name = "_"

            if op.USE_ID:
                append_name += op.USE_ID

            if op.IS_CLOSED:
                append_name += "closed_"

            file_name = DUMP_PATH + DUMP_FILE + append_name + self.file_name + "_" + str(op.RATIO)

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
