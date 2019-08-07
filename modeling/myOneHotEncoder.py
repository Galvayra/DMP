# -*- coding: utf-8 -*-
from collections import OrderedDict
from .variables import *
from DMP.modeling.w2vReader import W2vReader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from os import listdir
import math
import numpy as np
import json

COLUMN_NUMBER = 'A'


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder(W2vReader):
    def __init__(self, ver, log_name, num_of_important, ct_image_path=""):
        super().__init__()
        self.__vector = OrderedDict()

        # a dictionary for encoder when 'fit' function called
        self.__vector_dict = dict()

        # a encoded vector matrix to transform
        self.__vector_matrix = OrderedDict()

        # init handler for making dictionary
        self.dataHandler = None
        self.__x_data_dict = None
        self.__version = ver
        self.__importance = dict()
        self.__num_of_important = num_of_important

        if self.version == 1:
            print("\n========= Version is Making vector for training!! =========\n\n")
            self.__set_importance(log_name)
        elif self.version == 2:
            print("\n========= Version is Making vector for Feature Selection!! =========\n\n")

        if USE_QUANTIZATION:
            print("Using Vector Quantization!!\n")
        elif USE_STANDARD_SCALE:
            print("Using Standard Scale!!\n")
        else:
            print("Using MinMax Scale!!\n")

        # { 0: ["column", "header"], .... n: ["column", "header"] }
        # a dictionary to show how to match feature to dimensionality
        self.__feature_dict = dict()
        self.__num_of_dim = int()

        self.__ct_image_path = ct_image_path

    @property
    def vector(self):
        return self.__vector

    @property
    def vector_dict(self):
        return self.__vector_dict

    @property
    def vector_matrix(self):
        return self.__vector_matrix

    @vector_matrix.setter
    def vector_matrix(self, vector_matrix):
        self.__vector_matrix = vector_matrix

    @property
    def x_data_dict(self):
        return self.__x_data_dict

    @property
    def feature_dict(self):
        return self.__feature_dict

    @property
    def importance(self):
        return self.__importance

    @property
    def num_of_important(self):
        return self.__num_of_important

    @property
    def num_of_dim(self):
        return self.__num_of_dim

    @num_of_dim.setter
    def num_of_dim(self, num_of_dim):
        self.__num_of_dim = num_of_dim

    @property
    def version(self):
        return self.__version

    @property
    def ct_image_path(self):
        return self.__ct_image_path

    def __set_importance(self, log_name):
        if log_name:
            with open(log_name, 'r') as r_file:
                for k, v in json.load(r_file).items():
                    self.importance[k] = v
                print("success load file! - file name is", log_name)
                print("\n=========================================================\n\n")

    # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
    @staticmethod
    def __set_scalar_dict(value_list):
        scalar_dict = dict()
        scalar_list = list()

        # for v in sorted(list(set(value_list))):
        #     # 공백은 사전에 넣지 않음
        #     if not math.isnan(v):
        #         scalar_list.append(v)

        for v in sorted(value_list):
            # 공백은 사전에 넣지 않음
            if not math.isnan(v):
                scalar_list.append(v)

        if scalar_list:
            scalar_dict["list"] = scalar_list
            scalar_dict["max"] = max(scalar_list)
            scalar_dict["min"] = min(scalar_list)
            scalar_dict["dif"] = float(scalar_dict["max"] - scalar_dict["min"])

        # print("\n" + column)
        # # print(scalar_list)
        # print(scalar_dict)
        # print()

        return scalar_dict

    # 셀의 공백은 type is not str 으로 찾을 수 있으며, 공백(nan)을 하나의 차원으로 볼지에 대한 선택을 우선 해야한다
    @staticmethod
    def __set_class_dict(value_list):
        class_dict = dict()

        for v in value_list:
            # key exception is nan
            if v != "nan":
                if v not in class_dict:
                    class_dict[v] = 1
                else:
                    class_dict[v] += 1

        return class_dict

    def __set_one_hot_dict(self, value_list):
        one_hot_dict = dict()

        if self.version == 1:
            for v in value_list:
                # key exception is nan
                if v != "nan":
                    v = v.split('_')
                    for token in v:
                        if token not in one_hot_dict:
                            one_hot_dict[token] = 1
                        else:
                            one_hot_dict[token] += 1
        # for feature selection
        elif self.version == 2:
            for v in value_list:
                if v != "nan":
                    if v not in one_hot_dict:
                        one_hot_dict[v] = 1
                    else:
                        one_hot_dict[v] += 1

        # print(column.ljust(30), len(one_hot_dict))
        # print("-----------------------------------------------------------")
        #
        # for token in sorted(one_hot_dict.items(), key=lambda x: x[1], reverse=True):
        #     print(token[0].ljust(30), token[1])
        # print("\n\n")

        return one_hot_dict

    def __set_embedded_dict(self, value_list):
        embedded_dict = dict()

        if self.w2v_dict:
            for v in value_list:
                # key exception is nan
                if v != "nan":
                    v = v.split('_')
                    for token in v:
                        if self.has_key_in_w2v_dict(token):
                            if token not in embedded_dict:
                                embedded_dict[token] = 1
                            else:
                                embedded_dict[token] += 1

        # print(column.ljust(30), len(embedded_dict))
        # print("-----------------------------------------------------------")
        #
        # for token in sorted(embedded_dict.items(), key=lambda x: x[1], reverse=True):
        #     print(token[0].ljust(30), token[1])
        # print("\n\n")

        return embedded_dict

    def __set_feature_dict(self, column_info, length):
        for i in range(length):
            self.feature_dict[self.num_of_dim] = column_info
            self.num_of_dim += 1

    def __get_column_list(self):
        column_list = list()

        if self.importance:
            column_import_list = [c for c in self.importance]

            for column in list(self.x_data_dict.keys()):
                if column in column_import_list[:self.num_of_important]:
                    column_list.append(column)
        else:
            for column in list(self.x_data_dict.keys()):
                column_list.append(column)

        return column_list

    def fit(self, data_handler):
        # init handler for making dictionary
        self.dataHandler = data_handler
        self.__x_data_dict = data_handler.x_data_dict

        for column in self.__get_column_list():
            type_of_column = self.dataHandler.get_type_of_column(column)
            column_info = [column, self.dataHandler.raw_header_dict[column]]

            if type_of_column == "id":
                continue
            # type of column is "scalar"
            elif type_of_column == "scalar":
                self.vector_dict[column] = self.__set_scalar_dict(self.x_data_dict[column])

                if self.version == 1:
                    self.__set_feature_dict(column_info, len(SCALAR_VECTOR))
                elif self.version == 2:
                    self.__set_feature_dict(column_info, 1)
            # type of column is "class"
            elif type_of_column == "class":
                self.vector_dict[column] = self.__set_class_dict(self.x_data_dict[column])

                if self.version == 1:
                    self.__set_feature_dict(column_info, len(self.vector_dict[column]))
                elif self.version == 2:
                    self.__set_feature_dict(column_info, 1)
            # type of column is "symptom" or "mal_type" or "diagnosis"
            elif type_of_column == "symptom" or type_of_column == "mal_type" or type_of_column == "diagnosis":

                if self.version == 1:
                    if self.w2v_dict:
                        self.vector_w2v_dict[column] = self.__set_embedded_dict(self.x_data_dict[column])
                        self.__set_feature_dict(column_info, self.dimension)

                        # Extended word vector = < w2v + one_hot >
                        if EXTENDED_WORD_VECTOR:
                            self.vector_dict[column] = self.__set_one_hot_dict(self.x_data_dict[column])
                            self.__set_feature_dict(column_info, len(self.vector_dict[column]))
                    else:
                        self.vector_dict[column] = self.__set_one_hot_dict(self.x_data_dict[column])
                        self.__set_feature_dict(column_info, len(self.vector_dict[column]))
                elif self.version == 2:
                    self.vector_dict[column] = self.__set_one_hot_dict(self.x_data_dict[column])
                    self.__set_feature_dict(column_info, 1)
            # type of column is "word"
            elif type_of_column == "word":
                self.vector_dict[column] = self.__set_one_hot_dict(self.x_data_dict[column])

                if self.version == 1:
                    self.__set_feature_dict(column_info, len(self.vector_dict[column]))
                elif self.version == 2:
                    self.__set_feature_dict(column_info, 1)

    def get_feature_dict(self):
        return self.feature_dict

    @staticmethod
    def __get_data_list(target_list):
        data_list = list()

        for x in target_list:
            if math.isnan(x):
                data_list.append([0.0])
            else:
                data_list.append([x])

        return data_list

    # make a generator for scalar vector
    def __set_scalar_vector(self, column, target_data_dict, *scale):
        def __get_quantize_dict():
            _quantize_dict = dict()
            div_range = int(len(scalar_list) / NUM_QUANTIZE)

            for value in range(1, NUM_QUANTIZE):
                _quantize_dict[value] = scalar_list[div_range * value]

            return _quantize_dict

        def __get_quantize_value():
            for k, v in quantize_dict.items():
                if target_value <= v:
                    return [k / NUM_QUANTIZE]

            return [1.0]

        # If dict of scalar vector, make vector using dict
        # But, If not have it, do not make vector (we consider that the column will be noise)
        if self.vector_dict[column]:
            data_list = self.__get_data_list(target_data_dict[column])
            vector_list = list()

            if self.version == 1:
                # #### using function scaling version
                # scaling
                if scale[0] is not None:
                    for i in range(len(scale)):
                        scale[i].fit(data_list)
                        data_list = scale[i].transform(data_list)

                    # processing 'nan' value after transform
                    for x in data_list:
                        if math.isnan(x):
                            vector_list.append([0.0])
                        else:
                            vector_list.append(x)

                    # copy values into the vector matrix
                    for index, vector in enumerate(vector_list):
                        yield index, vector

                # vector quantization
                else:
                    scalar_list = self.vector_dict[column]["list"]
                    quantize_dict = __get_quantize_dict()

                    # copy values into the vector matrix
                    for index, vector in enumerate(data_list):
                        target_value = vector[0]

                        if math.isnan(target_value):
                            vector = [0.0]
                        else:
                            vector = __get_quantize_value()

                        yield index, vector

                # # ##### original scaling version
                # differ = self.vector_dict[column]["dif"]
                # minimum = self.vector_dict[column]["min"]
                #
                # # The differ is 0 == The scalar vector size is 1
                # # ex) vector size == 1
                # #     if value in vector_dict ? [1.0] : [0.0]
                # if not differ:
                #     for index, data in enumerate(data_list):
                #         vector = [0.0]
                #         value = data[0]
                #
                #         if not math.isnan(value):
                #             vector[0] = 1.0
                #
                #         vector_list.append(vector)
                #
                # # ex) vector size > 1
                # #     if value in vector_dict ? [SCALAR_DEFAULT_WEIGHT, value] : [0.0, 0.0]
                # else:
                #     for index, data in enumerate(data_list):
                #         vector = SCALAR_VECTOR[:]
                #         value = data[0]
                #
                #         if not math.isnan(value):
                #             if len(vector) == 2:
                #                 vector[0] = SCALAR_DEFAULT_WEIGHT
                #                 vector[1] = (value - minimum) / differ
                #             else:
                #                 vector[0] = (value - minimum) / differ
                #
                #         vector_list.append(vector)
                #
                # # copy values into the vector matrix
                # for index, vector in enumerate(vector_list):
                #     yield index, vector

            elif self.version == 2:
                # processing 'nan' value after transform
                for x in data_list:
                    if math.isnan(x[0]):
                        vector_list.append([0.0])
                    else:
                        vector_list.append(x)

                # copy values into the vector matrix
                for index, vector in enumerate(vector_list):
                    yield index, vector

    # make a generator for class vector
    def __set_class_vector(self, column, target_data_dict):
        for index, value in enumerate(target_data_dict[column]):
            yield index, self.__get_one_hot([value], self.vector_dict[column])

    # make a generator for embedded vector
    def __set_embedded_vector(self, column, target_data_dict):
        for index, value in enumerate(target_data_dict[column]):
            yield index, self.get_w2v_vector(value.split('_'), column)

    # make a generator for one-hot vector
    def __set_one_hot_vector(self, column, target_data_dict):
        if self.version == 1:
            for index, value in enumerate(target_data_dict[column]):
                yield index, self.__get_one_hot(value.split('_'), self.vector_dict[column])
        elif self.version == 2:
            for index, value in enumerate(target_data_dict[column]):
                yield index, self.__get_one_hot([value], self.vector_dict[column])

    def __get_one_hot(self, word, vector_dict):
        one_hot_vector = list()

        for w in vector_dict:
            if w in word:
                one_hot_vector.append(float(1))
            else:
                one_hot_vector.append(float(0))

        if self.version == 1:
            return one_hot_vector
        elif self.version == 2:
            return [float(np.argmax(one_hot_vector))]

    # set vector into self.vector_matrix using generator
    def __set_vector(self, class_of_column, generator):
        if generator:
            if USE_CLASS_VECTOR:
                for index, vector in generator:
                    for v in vector:
                        self.vector_matrix[KEY_NAME_OF_MERGE_VECTOR][index].append(v)
                        self.vector_matrix[class_of_column][index].append(v)
            else:
                for index, vector in generator:
                    for v in vector:
                        self.vector_matrix[KEY_NAME_OF_MERGE_VECTOR][index].append(v)

    def __init_vector_matrix(self, data_handler):
        self.vector_matrix = OrderedDict()

        self.vector_matrix[KEY_NAME_OF_MERGE_VECTOR] = list()
        num_of_data = len(data_handler.y_data)

        for _ in range(num_of_data):
            self.vector_matrix[KEY_NAME_OF_MERGE_VECTOR].append(list())

        # set X(number of rows) using rows_data
        if USE_CLASS_VECTOR:
            for class_of_column in self.dataHandler.columns_dict:
                self.vector_matrix[class_of_column] = list()

                for _ in range(num_of_data):
                    self.vector_matrix[class_of_column].append(list())

    def transform2matrix(self, data_handler):
        self.__init_vector_matrix(data_handler)
        self.__transform(data_handler=data_handler)

        return self.vector_matrix

    def __transform(self, data_handler):
        target_data_dict = data_handler.x_data_dict

        if self.version == 1:
            for column in self.__get_column_list():
                type_of_column = self.dataHandler.get_type_of_column(column)
                class_of_column = self.dataHandler.get_class_of_column(column)
                generator = False

                if type_of_column == "id":
                    pass
                elif type_of_column == "scalar":
                    # using standard scaling
                    if USE_QUANTIZATION:
                        generator = self.__set_scalar_vector(column, target_data_dict, None)
                    elif USE_STANDARD_SCALE:
                        generator = self.__set_scalar_vector(column, target_data_dict, StandardScaler())
                    # using min max scaling
                    else:
                        generator = self.__set_scalar_vector(column, target_data_dict, MinMaxScaler())
                elif type_of_column == "class":
                    generator = self.__set_class_vector(column, target_data_dict)
                elif type_of_column == "symptom" or type_of_column == "mal_type" or type_of_column == "diagnosis":
                    if self.w2v_dict:
                        generator = self.__set_embedded_vector(column, target_data_dict)

                        if EXTENDED_WORD_VECTOR:
                            self.__set_vector(class_of_column, generator)
                            generator = self.__set_one_hot_vector(column, target_data_dict)
                    else:
                        generator = self.__set_one_hot_vector(column, target_data_dict)
                elif type_of_column == "word":
                    generator = self.__set_one_hot_vector(column, target_data_dict)

                self.__set_vector(class_of_column, generator)

        elif self.version == 2:
            for column in self.__get_column_list():
                type_of_column = self.dataHandler.get_type_of_column(column)
                class_of_column = self.dataHandler.get_class_of_column(column)
                generator = False

                if type_of_column == "id":
                    pass
                elif type_of_column == "scalar":
                    if USE_QUANTIZATION:
                        generator = self.__set_scalar_vector(column, target_data_dict, None)
                    elif USE_STANDARD_SCALE:
                        generator = self.__set_scalar_vector(column, target_data_dict, StandardScaler())
                    else:
                        generator = self.__set_scalar_vector(column, target_data_dict, MinMaxScaler())
                elif type_of_column == "class":
                    generator = self.__set_class_vector(column, target_data_dict)
                elif type_of_column == "symptom" or \
                        type_of_column == "mal_type" or \
                        type_of_column == "diagnosis" or \
                        type_of_column == "word":
                    generator = self.__set_one_hot_vector(column, target_data_dict)

                self.__set_vector(class_of_column, generator)

    def transform2image_matrix(self, data_handler):
        image_matrix = list()
        target_data_dict = data_handler.x_data_dict

        for p_number in target_data_dict['A'].values():
            target_path = self.ct_image_path + p_number + "/"

            patient_image_matrix = list()

            for image_name in sorted(listdir(target_path)):
                target_image_path = target_path + image_name
                patient_image_matrix.append(target_image_path)

            image_matrix.append(patient_image_matrix)

        return image_matrix

    def show_vectors(self, *columns):
        for i, column in enumerate(columns):
            if column in self.vector_dict:
                print("column " + str(i + 1) + " -", column, "\n")
                print(self.vector_dict[column], "\n")
                for data, data_vector in zip(self.x_data_dict[column], self.vector[column]):
                    print(str(data))
                    print(data_vector)
                print("\n=============================================================\n\n")
