# -*- coding: utf-8 -*-
from collections import OrderedDict
from .variables import *
from DMP.modeling.w2vReader import W2vReader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder(W2vReader):
    def __init__(self, data_handler, ver):
        super().__init__()
        self.__vector = OrderedDict()
        self.__vector_dict = dict()

        # init handler for making dictionary
        self.dataHandler = data_handler
        self.__x_data_dict = data_handler.x_data_dict

    @property
    def vector(self):
        return self.__vector

    @property
    def vector_dict(self):
        return self.__vector_dict

    @property
    def x_data_dict(self):
        return self.__x_data_dict

    def encoding(self):
        # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
        def __set_scalar_dict(value_list):
            scalar_dict = dict()
            scalar_list = list()

            for v in sorted(list(set(value_list))):
                # 공백은 사전에 넣지 않음
                if not math.isnan(v):
                    scalar_list.append(v)

            if scalar_list:
                scalar_dict["max"] = max(scalar_list)
                scalar_dict["min"] = min(scalar_list)
                scalar_dict["dif"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print("\n" + column)
            # # print(scalar_list)
            # print(scalar_dict)
            # print()

            return scalar_dict

        # 셀의 공백은 type is not str 으로 찾을 수 있으며, 공백(nan)을 하나의 차원으로 볼지에 대한 선택을 우선 해야한다
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

        def __set_one_hot_dict(value_list):
            one_hot_dict = dict()

            for v in value_list:
                # key exception is nan
                if v != "nan":
                    v = v.split('_')
                    for token in v:
                        if token not in one_hot_dict:
                            one_hot_dict[token] = 1
                        else:
                            one_hot_dict[token] += 1

            # print(column.ljust(30), len(one_hot_dict))
            # print("-----------------------------------------------------------")
            #
            # for token in sorted(one_hot_dict.items(), key=lambda x: x[1], reverse=True):
            #     print(token[0].ljust(30), token[1])
            # print("\n\n")

            return one_hot_dict

        def __set_embedded_dict(value_list):
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

        # { 0: ["column", "header"], .... n: ["column", "header"] }
        # n == dimensionality
        feature_dict = dict()
        dimensionality = int()

        for column in list(self.x_data_dict.keys()):
            type_of_column = self.dataHandler.get_type_of_column(column)
            column_info = [column, self.dataHandler.raw_header_dict[column]]

            if type_of_column == "id":
                continue
            # type of column is "scalar"
            elif type_of_column == "scalar":
                self.vector_dict[column] = __set_scalar_dict(self.x_data_dict[column])

                for _ in range(len(SCALAR_VECTOR)):
                    feature_dict[dimensionality] = column_info
                    dimensionality += 1
            # type of column is "class"
            elif type_of_column == "class":
                self.vector_dict[column] = __set_class_dict(self.x_data_dict[column])

                for _ in range(len(self.vector_dict[column])):
                    feature_dict[dimensionality] = column_info
                    dimensionality += 1
            # type of column is "symptom" or "mal_type" or "diagnosis"
            elif type_of_column == "symptom" or type_of_column == "mal_type" or type_of_column == "diagnosis":
                if self.w2v_dict:
                    self.vector_w2v_dict[column] = __set_embedded_dict(self.x_data_dict[column])

                    for _ in range(self.dimension):
                        feature_dict[dimensionality] = column_info
                        dimensionality += 1

                    # Extended word vector = < w2v + one_hot >
                    if EXTENDED_WORD_VECTOR:
                        self.vector_dict[column] = __set_one_hot_dict(self.x_data_dict[column])

                        for _ in range(len(self.vector_dict[column])):
                            feature_dict[dimensionality] = column_info
                            dimensionality += 1
                else:
                    self.vector_dict[column] = __set_one_hot_dict(self.x_data_dict[column])

                    for _ in range(len(self.vector_dict[column])):
                        feature_dict[dimensionality] = column_info
                        dimensionality += 1
            # type of column is "word"
            elif type_of_column == "word":
                self.vector_dict[column] = __set_one_hot_dict(self.x_data_dict[column])

                for _ in range(len(self.vector_dict[column])):
                    feature_dict[dimensionality] = column_info
                    dimensionality += 1

        return feature_dict

    def __init_vector(self, data_handler):
        vector_matrix = OrderedDict()

        vector_matrix[KEY_NAME_OF_MERGE_VECTOR] = list()
        num_of_data = len(data_handler.y_data)

        for _ in range(num_of_data):
            vector_matrix[KEY_NAME_OF_MERGE_VECTOR].append(list())

        # set X(number of rows) using rows_data
        for class_of_column in self.dataHandler.columns_dict:
            vector_matrix[class_of_column] = list()

            for _ in range(num_of_data):
                vector_matrix[class_of_column].append(list())

        return vector_matrix

    def fit(self, data_handler):
        def __set_scalar_vector(*scale):
            # If dict of scalar vector, make vector using dict
            # But, If not have it, do not make vector (we consider that the column will be noise)
            if self.vector_dict[column]:

                # ##### using function scaling version
                data_list = [[x] for x in target_data_dict[column]]

                # scaling
                for i in range(len(scale)):
                    scale[i].fit(data_list)
                    data_list = scale[i].transform(data_list)

                # processing 'nan' value
                values = list()
                for x in data_list:
                    if math.isnan(x):
                        values.append([0.0])
                    else:
                        values.append(x)

                # copy values into the vector matrix
                for index, value in enumerate(values):
                    __set_vector(index, value)

                # # ##### using function scaling version if exist feature has vector size == 1
                # differ = self.vector_dict[column]["dif"]
                #
                # # The differ is 0 == The scalar vector size is 1
                # # ex) vector size == 1
                # #     if value in vector_dict ? [1.0] : [0.0]
                # if not differ:
                #     for index, value in enumerate(target_data_dict[column]):
                #
                #         if math.isnan(value):
                #             values = [0.0]
                #         else:
                #             values = [1.0]
                #
                #         __set_vector(index, values)
                # # ex) vector size > 1
                # #     if value in vector_dict --> min max scaling
                # else:
                #     data_list = [[x] for x in target_data_dict[column]]
                #
                #     for i in range(len(scale)):
                #         scale[i].fit(data_list)
                #         data_list = scale[i].transform(data_list)
                #
                #     values = list()
                #     for x in data_list:
                #         if math.isnan(x):
                #             values.append([0.0])
                #         else:
                #             values.append(x)
                #
                #     for index, value in enumerate(values):
                #         __set_vector(index, value)

                # # ##### original scaling version
                # differ = self.vector_dict[column]["dif"]
                # minimum = self.vector_dict[column]["min"]
                #
                # # The differ is 0 == The scalar vector size is 1
                # # ex) vector size == 1
                # #     if value in vector_dict ? [1.0] : [0.0]
                # if not differ:
                #     for index, value in enumerate(target_data_dict[column]):
                #         values = [0.0]
                #
                #         if not math.isnan(value):
                #             values[0] = 1.0
                #
                #         __set_vector(index, values)
                # # ex) vector size > 1
                # #     if value in vector_dict ? [SCALAR_DEFAULT_WEIGHT, value] : [0.0, 0.0]
                # else:
                #     for index, value in enumerate(target_data_dict[column]):
                #         values = SCALAR_VECTOR[:]
                #
                #         if not math.isnan(value):
                #             if len(values) == 2:
                #                 values[0] = SCALAR_DEFAULT_WEIGHT
                #                 values[1] = (value - minimum) / differ
                #             else:
                #                 values[0] = (value - minimum) / differ
                #
                #         __set_vector(index, values)

        def __set_class_vector():
            for index, value in enumerate(target_data_dict[column]):
                __set_vector(index, __get_one_hot([value], self.vector_dict[column]))

        def __set_embedded_vector():
            for index, value in enumerate(target_data_dict[column]):
                __set_vector(index, self.get_w2v_vector(value.split('_'), column))

        def __set_one_hot_vector():
            for index, value in enumerate(target_data_dict[column]):
                __set_vector(index, __get_one_hot(value.split('_'), self.vector_dict[column]))

        def __get_one_hot(word, vector_dict):
            one_hot_vector = list()

            for w in vector_dict:
                if w in word:
                    one_hot_vector.append(float(1))
                else:
                    one_hot_vector.append(float(0))

            return one_hot_vector

        def __set_vector(index, vector):
            for v in vector:
                vector_matrix[KEY_NAME_OF_MERGE_VECTOR][index].append(v)
                vector_matrix[class_of_column][index].append(v)

        vector_matrix = self.__init_vector(data_handler)
        target_data_dict = data_handler.x_data_dict

        for column in list(self.x_data_dict.keys()):
            type_of_column = self.dataHandler.get_type_of_column(column)
            class_of_column = self.dataHandler.get_class_of_column(column)

            if type_of_column == "id":
                pass
            elif type_of_column == "scalar":
                # using standard scaling
                if USE_STANDARD_SCALE:
                    __set_scalar_vector(StandardScaler())
                else:
                    __set_scalar_vector(MinMaxScaler())
            elif type_of_column == "class":
                __set_class_vector()
            elif type_of_column == "symptom" or type_of_column == "mal_type" or type_of_column == "diagnosis":
                if self.w2v_dict:
                    __set_embedded_vector()

                    if EXTENDED_WORD_VECTOR:
                        __set_one_hot_vector()
                else:
                    __set_one_hot_vector()
            elif type_of_column == "word":
                __set_one_hot_vector()

        return vector_matrix

    def show_vectors(self, *columns):
        for i, column in enumerate(columns):
            if column in self.vector_dict:
                print("column " + str(i + 1) + " -", column, "\n")
                print(self.vector_dict[column], "\n")
                for data, data_vector in zip(self.x_data_dict[column], self.vector[column]):
                    print(str(data))
                    print(data_vector)
                print("\n=============================================================\n\n")
