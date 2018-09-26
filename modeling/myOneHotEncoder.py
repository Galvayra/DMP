# -*- coding: utf-8 -*-
import math
import gensim.models.keyedvectors as word2vec
from .variables import *

DIMENSION_W2V = 300
MIN_SCALING = 0.1


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder:
    def __init__(self, data_handler, w2v=False):
        self.__vector = OrderedDict()
        self.__vector_dict = dict()
        self.dataHandler = data_handler
        self.__x_data = self.dataHandler.x_data_dict
        # self.__vector_dict = dict()
        # self.__w2v = w2v
        # if self.w2v:
        #     self.model = word2vec.KeyedVectors.load_word2vec_format(DUMP_PATH + LOAD_WORD2VEC, binary=True)
        #     print("\nUsing word2vec")
        #     print("\nRead w2v file -", DUMP_PATH + LOAD_WORD2VEC)
        # else:
        #     print("\nNot using Word2vec")
        # print("\n\n")

    @property
    def vector(self):
        return self.__vector

    @property
    def vector_dict(self):
        return self.__vector_dict

    @property
    def x_data(self):
        return self.__x_data

    #
    # @property
    # def w2v(self):
    #     return self.__w2v

    def encoding(self):

        # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
        def __set_scalar_dict(value_list):
            scalar_dict = dict()
            scalar_list = list()

            for v in sorted(list(set(value_list))):
                # 공백은 사전에 넣지 않음
                if not math.isnan(v):
                    scalar_list.append(v)

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print("\n" + header)
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

            # print(header.ljust(30), len(one_hot_dict))
            # print("-----------------------------------------------------------")
            #
            # for token in sorted(one_hot_dict.items(), key=lambda x: x[1], reverse=True):
            #     print(token[0].ljust(30), token[1])
            # print("\n\n")

            return one_hot_dict

        for column in list(self.x_data.keys()):
            type_of_column = self.dataHandler.get_type_of_column(column)

            if type_of_column == "id":
                continue
            elif type_of_column == "scalar":
                self.vector_dict[column] = __set_scalar_dict(self.x_data[column])
            elif type_of_column == "class":
                self.vector_dict[column] = __set_class_dict(self.x_data[column])
            else:
                self.vector_dict[column] = __set_one_hot_dict(self.x_data[column])

    def __init_vector(self):
        # _x_vector_dict = OrderedDict()
        self.vector[KEY_NAME_OF_MERGE_VECTOR] = list()

        for _ in range(len(self.dataHandler.y_data)):
            self.vector[KEY_NAME_OF_MERGE_VECTOR].append(list())

        # set X(number of rows) using rows_data
        for class_of_column in self.dataHandler.columns_dict:
            self.vector[class_of_column] = list()

            for _ in range(len(self.dataHandler.y_data)):
                self.vector[class_of_column].append(list())

    def fit2(self, num_fold):

        self.__init_vector()

        # for k in self.vector:
        #     print(k, len(self.vector[k]))

        for class_of_column, _ in self.dataHandler.columns_dict.items():
            for type_of_column, column_of_list in _.items():
                for column in column_of_list:
                    self.__vector_maker(class_of_column, column, type_of_column)

                        # self.vector_dict[header] = __set_vector(x_data[header])
                    # elif type_of_column == "class":
                    #     self.vector_dict[header] = __set_class_dict(x_data[header])
                    # else:
                    #     self.vector_dict[header] = __set_one_hot_dict(x_data[header])

    def __vector_maker(self, class_of_column, column, type_of_column):
        def __set_scalar_vector():
            minimum = self.vector_dict[column]["max"]
            division = self.vector_dict[column]["div"]

            for index, value in enumerate(self.x_data[column]):

                # scalar vector ex) [exist value]
                # if exist is 0 == The value is NaN
                # if exist is 1 == The value is existed
                values = [0.0, 0.0]

                # The value is NaN
                if not math.isnan(value):
                    values[0] = 1.0
                    values[1] = (value - minimum + MIN_SCALING) / (division + MIN_SCALING)

                __set_vector(index, values)

        def __set_class_vector():
            # print(column, self.vector_dict[column])
            for index, value in enumerate(self.x_data[column]):
                __set_vector(index, __get_one_hot([value], self.vector_dict[column]))

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
                self.vector[KEY_NAME_OF_MERGE_VECTOR][index].append(v)
                self.vector[class_of_column][index].append(v)

        if type_of_column == "id":
            return
        elif type_of_column == "scalar":
            __set_scalar_vector()
        elif type_of_column == "class":
            __set_class_vector()

    def fit(self, data_dict, data_count):
        def __init_x_vector_dict():
            # _x_vector_dict = OrderedDict()
            _x_vector_dict = OrderedDict()
            _x_vector_dict[KEY_NAME_OF_MERGE_VECTOR] = list()

            for _columns_key in columns_dict:
                _x_vector_dict[_columns_key] = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _key in _x_vector_dict:
                for _i in range(data_count):
                    _x_vector_dict[_key].append(list())

            return _x_vector_dict

        def __make_vector_use_scalar():
            value_list = self.__set_scalar_value_list(k, v)

            for _i, _value in enumerate(value_list):
                # type is float
                if math.isnan(_value):
                    _value = float(0)
                elif _value < minimum:
                    _value = float(0)
                elif _value > maximum:
                    _value = float(1)
                # normalization
                else:
                    # print(_value, (_value - minimum + MIN_SCALING)/(division + MIN_SCALING))
                    _value = (_value - minimum + MIN_SCALING)/(division + MIN_SCALING)
                    # _value = (_value - minimum) / division

                x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][_i].append(_value)
                x_vector_dict[columns_key][_i].append(_value)
                self.vector[k].append(_value)

        def __make_vector_use_class():
            _value = str(value).strip()
            self.vector[k].append(list())

            if self.__is_zero(_value):
                _value = str(0)

            for c in class_list:
                if c == _value:
                    x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][i].append(float(1))
                    x_vector_dict[columns_key][i].append(float(1))
                    self.vector[k][i].append(float(1))
                else:
                    x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][i].append(float(0))
                    x_vector_dict[columns_key][i].append(float(0))
                    self.vector[k][i].append(float(0))

        def __make_one_hot(_word_list):
            for _c in class_list:
                if _c in _word_list:
                    x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][i].append(float(1))
                    x_vector_dict[columns_key][i].append(float(1))
                    self.vector[k][i].append(float(1))
                else:
                    x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][i].append(float(0))
                    x_vector_dict[columns_key][i].append(float(0))
                    self.vector[k][i].append(float(0))

        def __make_vector_use_word():
            self.vector[k].append(list())
            __make_one_hot(self.__get_word_list_culture(value))

        def __make_vector_use_mal_type():
            self.vector[k].append(list())
            __make_one_hot(self.__get_word_list_mal_type(value))

        def __make_vector_use_symptom():

            def __make_w2v_vector(x_vector):
                _div = len(w2v_vector_list)
                if _div > 0:
                    _vector = [0.0 for _ in range(len(w2v_vector_list[0]))]

                    for vector in w2v_vector_list:
                        for _index, _v in enumerate(vector):
                            _vector[_index] += _v

                    for _v in _vector:
                        x_vector.append(_v/_div)
                else:
                    for _v in range(DIMENSION_W2V):
                        x_vector.append(float(0))

            self.vector[k].append(list())
            _word_list = self.__get_word_list_symptom(value)

            if self.w2v:
                w2v_vector_list = list()

                for _word in _word_list:
                    try:
                        w2v_vector_list.append(self.model.wv[_word])
                    except KeyError:
                        pass

                __make_one_hot(_word_list)
                __make_w2v_vector(x_vector_dict[KEY_NAME_OF_MERGE_VECTOR][i])
                __make_w2v_vector(x_vector_dict[columns_key][i])
            else:
                __make_one_hot(_word_list)

        def __get_all_columns(_columns_dict):
            all_columns = list()
            for _columns in _columns_dict.values():
                all_columns += _columns

            return all_columns

        self.__init_vector(data_dict)
        x_vector_dict = __init_x_vector_dict()

        for k in data_dict:
            v = data_dict[k]
            for columns_key, columns in columns_dict.items():
                for columns_type_key in columns:
                    if columns_type_key == "scalar":
                        if k in __get_all_columns(columns[columns_type_key]):
                            encode_dict = self.vector_dict[k]
                            minimum = encode_dict["min"]
                            maximum = encode_dict["max"]
                            division = encode_dict["div"]
                            __make_vector_use_scalar()
                    elif columns_type_key == "class":
                        if k in columns[columns_type_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = sorted(encode_dict.keys())
                            for i, value in enumerate(v):
                                __make_vector_use_class()
                    elif columns_type_key == "word":
                        if k in columns[columns_type_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = sorted(encode_dict.keys())
                            for i, value in enumerate(v):
                                __make_vector_use_word()
                    elif columns_type_key == "symptom":
                        if k in columns[columns_type_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = sorted(encode_dict.keys())
                            for i, value in enumerate(v):
                                __make_vector_use_symptom()
                    elif columns_type_key == "mal_type":
                        if k in columns[columns_type_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = sorted(encode_dict.keys())
                            for i, value in enumerate(v):
                                __make_vector_use_mal_type()

        return x_vector_dict

    # def __init_vector(self, data_dict):
    #     for k in data_dict:
    #         self.vector[k] = list()

    def show_vectors(self, x_data_dict, *columns):
        for k in columns:
            for data, data_vector in zip(x_data_dict[k], self.vector[k]):
                print(str(data))
                print(data_vector)
