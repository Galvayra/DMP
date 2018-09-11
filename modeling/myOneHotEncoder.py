# -*- coding: utf-8 -*-
import math
import gensim.models.keyedvectors as word2vec
from .variables import *

DIMENSION_W2V = 300
MIN_SCALING = 0.1


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder:
    def __init__(self, w2v=True):
        self.__vector = dict()
        self.__vector_dict = dict()
        self.__w2v = w2v
        if self.w2v:
            self.model = word2vec.KeyedVectors.load_word2vec_format(DUMP_PATH + LOAD_WORD2VEC, binary=True)
            print("\nUsing word2vec")
            print("\nRead w2v file -", DUMP_PATH + LOAD_WORD2VEC)
        else:
            print("\nNot using Word2vec")

    @property
    def vector(self):
        return self.__vector

    @property
    def vector_dict(self):
        return self.__vector_dict

    @property
    def w2v(self):
        return self.__w2v

    def encoding(self, data_dict):
        def __inspect_columns():
            # show result of columns inspecting
            k_dict = dict()

            for _k, v in data_dict.items():
                type_dict = dict()

                for i, value in enumerate(data_dict[_k]):
                    key = 0
                    if type(value) is float:
                        if math.isnan(value):
                            continue
                        key = "float"
                    elif type(value) is str:
                        key = "str"
                    elif type(value) is int:
                        key = "int"

                    if key not in type_dict:
                        type_dict[key] = 1
                    else:
                        type_dict[key] += 1

                k_dict[_k] = type_dict

            for _k in sorted(k_dict.keys()):
                print(_k, k_dict[_k])

        # columns 이 전부 float 형에 어긋나는 행과 데이터를 출력
        def __inspect_float_column():
            for _k in sorted(data_dict.keys()):
                if _k in blood_count_columns['scalar']:
                    for _i, v in enumerate(data_dict[_k]):
                        try:
                            v = float(v)
                        except ValueError:
                            print(_k, v, "(", _i, ")")

        # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
        def __set_scalar_dict(value_list, except_start=0, except_end=0):
            scalar_dict = dict()
            scalar_list = list()

            for _i in sorted(list(set(value_list))):
                # 공백은 사전에 넣지 않음
                if not math.isnan(_i):
                    scalar_list.append(_i)

            scalar_list = scalar_list[except_start:]

            for _i in range(except_end):
                scalar_list.pop()

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print("\n" + k)
            # # print(scalar_list)
            # print(scalar_dict)
            # print()

            return scalar_dict

        # 셀의 공백은 type is not str 으로 찾을 수 있으며, 공백(nan)을 하나의 차원으로 볼지에 대한 선택을 우선 해야한다
        def __set_class_dict(vector_list):

            class_dict = dict()

            for _i, v in enumerate(vector_list):
                v = str(v).strip()

                # key exception is nan
                if v != "nan":
                    if self.__is_zero(v):
                        v = str(0)

                    if v not in class_dict:
                        class_dict[v] = 1
                    else:
                        class_dict[v] += 1

            # print(k, len(class_dict), class_dict)

            return class_dict

        def __set_word_dict(vector_list):
            def __add_dict():
                for word in word_list:
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1

            word_dict = dict()

            for i, line in enumerate(vector_list):
                word_list = self.__get_word_list_culture(line)
                __add_dict()

            # print(len(word_dict))
            # for dd in word_dict:
            #     print(dd, word_dict[dd])

            return word_dict

        def __set_mal_type_dict(vector_list):
            def __add_dict():
                for word in word_list:
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1

            word_dict = dict()

            for i, line in enumerate(vector_list):
                word_list = self.__get_word_list_mal_type(line)
                __add_dict()

            # print(len(word_dict))
            # for dd in word_dict:
            #     print(dd, word_dict[dd])

            return word_dict

        def __set_symptom_dict(vector_list):
            def __add_dict():
                for word in word_list:
                    if word not in symptom_dict:
                        symptom_dict[word] = 1
                    else:
                        symptom_dict[word] += 1

            symptom_dict = dict()

            for line in vector_list:
                word_list = self.__get_word_list_symptom(line)
                __add_dict()

            # print(len(symptom_dict))
            # for dd in symptom_dict:
            #     print(dd, symptom_dict[dd])

            return symptom_dict

        # __inspect_columns()
        # __inspect_float_column()

        def __set_vector_dict():
            def __get_scalar_key(_key):
                _key_list = _key.split('#')

                if len(_key_list) == 2:
                    _start = int(_key_list[0].split('_')[1])
                    _end = int(_key_list[1].split('_')[1])
                else:
                    _key_list = _key_list[0].split('_')

                    if len(_key_list) == 2:
                        if _key_list[0] == "start":
                            _start = int(_key_list[1])
                            _end = 0
                        else:
                            _start = 0
                            _end = int(_key_list[1])
                    else:
                        _start = 0
                        _end = 0

                return _start, _end

            if columns_key == "scalar":
                for scalar_key, scalar_columns in columns[columns_key].items():
                    start, end = __get_scalar_key(scalar_key)
                    if k in columns[columns_key][scalar_key]:
                        v_list = self.__set_scalar_value_list(k, data_dict[k])
                        self.vector_dict[k] = __set_scalar_dict(v_list, except_start=start, except_end=end)

            elif columns_key == "class":
                if k in columns[columns_key]:
                    self.vector_dict[k] = __set_class_dict(data_dict[k])

            elif columns_key == "symptom":
                if k in columns[columns_key]:
                    self.vector_dict[k] = __set_symptom_dict(data_dict[k])

            elif columns_key == "word":
                if k in columns[columns_key]:
                    self.vector_dict[k] = __set_word_dict(data_dict[k])

            elif columns_key == "mal_type":
                if k in columns[columns_key]:
                    self.vector_dict[k] = __set_mal_type_dict(data_dict[k])

        for k in data_dict:
            for columns in columns_dict.values():
                for columns_key in columns:
                    __set_vector_dict()

        # print("\n\n=== AD ===")
        # for k in sorted(self.vector_dict["AD"]):
        #     print(k, self.vector_dict["AD"][k])
        # print("\n\n")

        # for k in sorted(self.vector_dict["CF"]):
        #     print(k, self.vector_dict["CF"][k])
        # print("\n\n")
        # for k in sorted(self.vector_dict["CI"]):
        #     print(k, self.vector_dict["CI"][k])
        # print("\n\n")
        # for k in sorted(self.vector_dict["CL"]):
        #     print(k, self.vector_dict["CL"][k])

    # str 형데이터를 scalar(float) 로 변환
    def __set_scalar_value_list(self, _, value_list):

        new_value_list = list()

        for _i, value in enumerate(value_list):
            try:
                new_value_list.append(float(value))
            except ValueError:
                # 측정불가와 셀의 값이 0인것
                if value == "측정불가" or self.__is_zero(value):
                    new_value_list.append(float(0))
                else:
                    ch = value[0]
                    if ch == ">" or ch == "<":
                        new_value_list.append(float(value[1:]))
                    else:
                        if value.find("이상") > 0:
                            v = value.split("이상")[0].strip()
                            new_value_list.append(float(v))
                        elif value.find("이하") > 0:
                            v = value.split("이하")[0].strip()
                            new_value_list.append(float(v))
                        else:
                            v = value.split("(")[0].strip()
                            new_value_list.append(float(v))

        return new_value_list

    # get word from symptom
    @staticmethod
    def __get_word_list_symptom(line):
        def __parsing(_w):
            _w = _w.strip().lower()
            _w = _w.replace('.', '. ')
            _w = _w.replace('(', ' ')
            _w = _w.replace(')', ' ')
            _w = "_".join(_w.split())
            _w = "_" + _w + "_"
            _w = _w.replace('_abd._', '_abdominal_')
            _w = _w.replace('_lt._', '_left_')
            _w = _w.replace('_rt._', '_right_')
            _w = _w.replace('_avf_', '_angioplasty_fails_')
            _w = _w.replace('_ptbd_', '_percutaneous_transhepatic_biliary_drainage_')
            _w = _w.replace('_bp_', '_blood_pressure_')
            _w = _w.replace('_cbc_', '_complete_blood_count_')
            _w = _w.replace('_ct_', '_computed_tomography_')
            _w = _w.replace('_lft_', '_liver_function_tests_')
            _w = _w.replace('_wbc_', '_white_blood_cell_')
            _w = _w.replace('_llq_', '_left_lower_quadrant_')
            _w = _w.replace('_luq_', '_left_upper_quadrant_')
            _w = _w.replace('_rlq_', '_right_lower_quadrant_')
            _w = _w.replace('_ruq_', '_right_upper_quadrant_')
            _w = _w.replace('_ugi_', '_upper_gastrointestinal_')
            _w = _w.replace('_hd_cath._', '_hemodialysis_catheter_')
            _w = _w.replace('_cath._', '_catheter_')
            _w = _w.replace('_exam._', '_examination_')
            _w = _w.replace('_t-tube_', '_tracheostomy_tube_')
            _w = _w.replace('_l-tube_', '_levin_tube_')
            _w = _w.replace('_peg_tube_', '_percutaneous_endoscopic_gastrostomy_tube_')
            _w = _w.replace('_op_', '_postoperative_')

            return _w[1:-1].split('_')

        lines = line.split(',')
        parsing_data_list = list()
        if len(lines) == 1:
            w = lines[0]
            parsing_data_list = __parsing(w)
        else:
            for w in lines:
                parsing_data_list += __parsing(w)

        return list(set(parsing_data_list))

    # get word from culture
    @staticmethod
    def __get_word_list_culture(line):
        def __parsing(_w):
            if not _w:
                return ["0"]

            _w = _w.strip().lower()
            _w = "_" + _w + "_"
            _w = "_".join(_w.split())
            _w = "_".join(_w.split("/"))
            _w = "_".join(_w.split("->"))
            _w = "_".join(_w.split("-."))
            _w = _w.replace('&', '_')

            return _w[1:-1].split('_')

        if type(line) is float:
            return ["0"]
        elif line == "0":
            return ["0"]
        else:
            lines = line.split(',')
            parsing_data_list = list()

            if len(lines) == 1:
                w = lines[0]
                parsing_data_list = __parsing(w)
            else:
                for w in lines:
                    parsing_data_list += __parsing(w)

            return list(set(parsing_data_list))

    # get word from culture
    @staticmethod
    def __get_word_list_mal_type(line):
        def __parsing(_w):
            if not _w:
                return ["0"]

            _w = _w.strip().lower()
            _w = "_" + _w + "_"
            _w = "_".join(_w.split())
            _w = "_".join(_w.split("/"))
            _w = _w.replace('_ca._', '_ca_')

            return _w[1:-1].split('_')

        if type(line) is float:
            return ["0"]
        elif line == "0":
            return ["0"]
        else:
            lines = line.split(',')
            parsing_data_list = list()

            if len(lines) == 1:
                w = lines[0]
                parsing_data_list = __parsing(w)
            else:
                for w in lines:
                    parsing_data_list += __parsing(w)

            return list(set(parsing_data_list))

    # "..", "..." 등 을 확인
    @staticmethod
    def __is_zero(string):
        is_zero = True
        for ch in string:
            if ch != '.':
                is_zero = False
        return is_zero

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

    def __init_vector(self, data_dict):
        for k in data_dict:
            self.vector[k] = list()

    def show_vectors(self, x_data_dict, *columns):
        for k in columns:
            for data, data_vector in zip(x_data_dict[k], self.vector[k]):
                print(str(data))
                print(data_vector)
