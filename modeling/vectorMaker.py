import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH
import json
import copy
import random

RANDRANGE = 10


class VectorMaker:
    # must using DataParser or DataHandler
    def __init__(self, data_handler):
        self.dataHandler = data_handler
        self.__y_data = self.dataHandler.y_data
        self.__len_data = len(self.dataHandler.y_data)
        self.__vector_list = list()
        self.__index = {
            "train": list(),
            "test": list(),
            "valid": list()
        }
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
    def index(self):
        return self.__index

    @property
    def vector_list(self):
        return self.__vector_list

    @vector_list.setter
    def vector_list(self, vector_list):
        self.__vector_list = vector_list

    def encoding(self):
        def __init_vector_dict():
            vector_dict = OrderedDict()
            vector_dict["x_train"] = x_train
            vector_dict["y_train"] = y_train
            vector_dict["x_test"] = x_test
            vector_dict["y_test"] = y_test

            return vector_dict

        def __set_x_data(is_manual=False, is_test=False):
            x_data = copy.deepcopy(encoder.vector_matrix)

            if is_manual:
                if is_test:
                    for class_of_column in list(x_data.keys()):
                        x_data[class_of_column] = x_data[class_of_column][:subset_size]
                else:
                    for class_of_column in list(x_data.keys()):
                        x_data[class_of_column] = x_data[class_of_column][subset_size:]
            else:
                if is_test:
                    for class_of_column in list(x_data.keys()):
                        x_data[class_of_column] = x_data[class_of_column][i * subset_size:][:subset_size]
                else:
                    for class_of_column in list(x_data.keys()):
                        x_data[class_of_column] = x_data[class_of_column][:i * subset_size] + \
                                                  x_data[class_of_column][(i + 1) * subset_size:]

            return x_data

        # init encoder and fit it
        # encoder = MyOneHotEncoder(self.dataHandler, w2v=op.USE_W2V)
        # encoder.encoding()
        # encoder.fit()
        # encoder.show_vectors(*self.dataHandler.header_list)

        self.__set_index()

        # for key, vectors in encoder.vector_matrix.items():
        #     for i, v in enumerate(vectors):
        #         print(key, i)
        #         print(v)
        #     print("\n===============================\n")

        # # k-fold validation
        # if op.NUM_FOLDS > 1:
        #     subset_size = int(self.len_data / op.NUM_FOLDS) + 1
        #
        #     for i in range(op.NUM_FOLDS):
        #         y_train = self.y_data[:i * subset_size] + self.y_data[(i + 1) * subset_size:]
        #         y_test = self.y_data[i * subset_size:][:subset_size]
        #         x_train = __set_x_data()
        #         x_test = __set_x_data(is_test=True)
        #         self.vector_list.append(__init_vector_dict())
        #
        # # one fold
        # else:
        #     subset_size = int(self.len_data / op.RATIO)
        #     y_train = self.y_data[subset_size:]
        #     y_test = self.y_data[:subset_size]
        #
        #     x_train = __set_x_data(is_manual=True)
        #     x_test = __set_x_data(is_manual=True, is_test=True)
        #     self.vector_list.append(__init_vector_dict())

        del self.dataHandler

    def __set_index(self):
        cut_train_ratio = op.RATIO
        cut_test_ratio = cut_train_ratio + ((RANDRANGE - cut_train_ratio) / 2.0)

        for i in range(self.len_data):
            rand = random.randrange(RANDRANGE)

            if rand < op.RATIO:
                self.index["train"].append(i)
            else:
                if cut_test_ratio.is_integer():
                    if rand < cut_test_ratio:
                        self.index["test"].append(i)
                    else:
                        self.index["valid"].append(i)
                else:
                    pass

                # print(cut_test_ratio, )
                # is_test = random.choice([True, False])
                #
                # if is_test:
                #     self.index["test"].append(i)
                # else:
                #     self.index["valid"].append(i)

        for k in self.index:
            print(k, len(self.index[k]))

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
            json.dump(self.vector_list, outfile, indent=4)
            print("\nsuccess make dump file! - file name is", file_name)

        if do_show:
            for i, data in enumerate(self.vector_list):
                print()
                print("\nData Set", i+1)
                print("Train total count -", str(len(self.vector_list[i]["x_train"]["merge"])).rjust(4),
                      "\tmortality count -", str(__counting_mortality(self.vector_list[i]["y_train"])).rjust(4))
                print("Test  total count -", str(len(self.vector_list[i]["x_test"]["merge"])).rjust(4),
                      "\tmortality count -", str(__counting_mortality(self.vector_list[i]["y_test"])).rjust(4))
            print()
