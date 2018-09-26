import DMP.utils.arg_encoding as op
from .myOneHotEncoder import MyOneHotEncoder
from collections import OrderedDict
from .variables import DUMP_FILE, DUMP_PATH
import json


class VectorMaker:
    # must using DataParser or DataHandler
    def __init__(self, data_handler):
        self.dataHandler = data_handler
        self.__vector_list = list()
        self.__file_name = self.dataHandler.file_name.split('.')[0]

    @property
    def file_name(self):
        return self.__file_name

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

        def __set_x_data_dict(is_manual=False, is_test=False):
            x_dict = dict()

            if is_manual:
                if is_test:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[:subset_size]
                else:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[subset_size:]
            else:
                if is_test:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[i * subset_size:][:subset_size]
                else:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[:i * subset_size] + _vector_list[(i + 1) * subset_size:]

            return x_dict

        # # copy DataHandler to local variables
        # x_data_dict = self.dataHandler.x_data_dict
        # y_data = self.dataHandler.y_data

        # self.dataHandler.show_type_of_columns()

        # for header, vector_list in self.dataHandler.x_data_dict.items():
        #     print(header)
        #

        # init encoder and fit it
        my_encoder = MyOneHotEncoder(self.dataHandler, w2v=op.USE_W2V)
        my_encoder.encoding()
        my_encoder.fit2(op.NUM_FOLDS)

        self.vector_list = my_encoder.vector

        # # k-fold validation
        # if op.NUM_FOLDS > 1:
        #     subset_size = int(len(y_data) / op.NUM_FOLDS) + 1
        #
        #     if op.IS_CLOSED:
        #         for i in range(op.NUM_FOLDS):
        #             y_train = y_data[:i * subset_size] + y_data[(i + 1) * subset_size:]
        #             y_test = y_data[:i * subset_size] + y_data[(i + 1) * subset_size:]
        #             x_train = my_encoder.fit(__set_x_data_dict(), len(y_train))
        #             x_test = my_encoder.fit(__set_x_data_dict(), len(y_test))
        #             self.vector_list.append(__init_vector_dict())
        #     else:
        #         for i in range(op.NUM_FOLDS):
        #             y_train = y_data[:i * subset_size] + y_data[(i + 1) * subset_size:]
        #             y_test = y_data[i * subset_size:][:subset_size]
        #             x_train = my_encoder.fit(__set_x_data_dict(), len(y_train))
        #             x_test = my_encoder.fit(__set_x_data_dict(is_test=True), len(y_test))
        #             self.vector_list.append(__init_vector_dict())
        #
        # # one fold
        # else:
        #     subset_size = int(len(y_data) / op.RATIO)
        #     y_train = y_data[subset_size:]
        #     y_test = y_data[:subset_size]
        #     x_train = my_encoder.fit2(__set_x_data_dict(is_manual=True), len(y_train))
        #     my_encoder.show_vectors(__set_x_data_dict(is_manual=True), "AD")
        #
        #     x_test = my_encoder.fit2(__set_x_data_dict(is_manual=True, is_test=True), len(y_test))
        #     my_encoder.show_vectors(__set_x_data_dict(is_manual=True, is_test=True), "AD")
        #
        #     self.vector_list.append(__init_vector_dict())

        del self.dataHandler

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

            file_name = DUMP_PATH + DUMP_FILE + append_name + self.file_name + "_" + str(op.NUM_FOLDS)

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_list, outfile, indent=4)
            print("\nsuccess make dump file! - file name is", file_name)

        # if do_show:
        #     for i, data in enumerate(self.vector_list):
        #         print()
        #         print("\nData Set", i+1)
        #         print("Train total count -", str(len(self.vector_list[i]["x_train"]["merge"])).rjust(4),
        #               "\tmortality count -", str(__counting_mortality(self.vector_list[i]["y_train"])).rjust(4))
        #         print("Test  total count -", str(len(self.vector_list[i]["x_test"]["merge"])).rjust(4),
        #               "\tmortality count -", str(__counting_mortality(self.vector_list[i]["y_test"])).rjust(4))
        #     print()
