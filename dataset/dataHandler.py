# -*- coding: utf-8 -*-
import pandas as pd
import math
import re
import random
from .variables import *

HAVE_SYMPTOM = 1
SEED = 444


# ### refer to reference file ###
class DataHandler:
    def __init__(self, data_file, do_parsing=False, column_target=False, eliminate_target=False):
        try:
            self.__file_name = data_file
            print("Read csv file -", DATA_PATH + self.file_name, "\n\n")
            self.__raw_data = pd.read_csv(DATA_PATH + self.file_name)
        except FileNotFoundError:
            print("There is no file !!\n\n")
            exit(-1)

        self.__column_target = column_target
        self.__columns_dict = columns_dict

        # eliminate target column in column dict
        if eliminate_target and column_target:
            for class_of_column in list(self.columns_dict.keys()):
                for type_of_column in list(self.columns_dict[class_of_column].keys()):
                    if self.column_target in self.columns_dict[class_of_column][type_of_column]:
                        self.columns_dict[class_of_column][type_of_column].remove(self.column_target)

        # header of data using column_dict in variables.py
        # [ 'C', 'E', .... 'CZ' ]
        self.header_list = self.__set_header_list()

        # a dictionary of header in raw data (csv file)
        # { header: name of column }
        self.__raw_header_dict = {self.__get_head_dict_key(i): v for i, v in enumerate(self.raw_data)}

        # a length of data
        self.__x_data_count = int()
        self.__y_data_count = int()

        # a dictionary of data
        # { header: a dictionary of data }
        self.x_data_dict = self.__set_data_dict()
        self.__do_parsing = do_parsing
        self.__do_set_data = False

        # # eliminate target column in column dict
        # if eliminate_target and column_target and column_target in self.header_list:

        if do_parsing:
            # except for data which is not necessary
            # [ position 1, ... position n ]
            self.__erase_index_list = self.__init_erase_index_list()

            # print(self.__erase_index_list, len(self.__erase_index_list))
            # print(len(self.__erase_index_list))
            self.__apply_exception()

        # a data of y labels
        # [ y_1, y_2, ... y_n ]
        self.y_data = self.__set_labels()

        # print(self.header_list)
        # print(self.x_data_dict.keys())

    @property
    def file_name(self):
        return self.__file_name

    @property
    def column_target(self):
        return self.__column_target

    @property
    def columns_dict(self):
        return self.__columns_dict

    @property
    def raw_data(self):
        return self.__raw_data

    @property
    def raw_header_dict(self):
        return self.__raw_header_dict

    @property
    def erase_index_list(self):
        return self.__erase_index_list
    
    @property
    def x_data_count(self):
        return self.__x_data_count

    @x_data_count.setter
    def x_data_count(self, count):
        self.__x_data_count = count

    @property
    def y_data_count(self):
        return self.__y_data_count

    @y_data_count.setter
    def y_data_count(self, count):
        self.__y_data_count = count

    @property
    def do_parsing(self):
        return self.__do_parsing

    @property
    def do_set_data(self):
        return self.__do_set_data

    @do_set_data.setter
    def do_set_data(self, do_set_data):
        self.__do_set_data = do_set_data

    def __get_head_dict_key(self, index):

        alpha_dict = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
        }
        alpha_len = len(alpha_dict)
        key = str()

        if index < alpha_len:
            return alpha_dict[index]

        key_second = int(index / alpha_len) - 1
        key_first = index % alpha_len

        key += self.__get_head_dict_key(key_second)
        key += alpha_dict[key_first]

        return key

    # initialize header list using columns_dict in variables.py
    def __set_header_list(self):
        header_list = list()

        for _ in self.columns_dict.values():
            for __ in _.values():
                for column in __:
                    header_list.append(column)

        return header_list

    def __get_raw_data(self, header):
        return self.raw_data[self.raw_header_dict[header]]

    def __set_data_dict(self):
        # {
        #   column: { row: data }
        #   C: { 2: C_1, 3: C_2, ... n: C_n }       ## ID
        #   E: { .... }                             ## Age
        #   ...
        #   CZ: { .... }                            ## Final Diagnosis
        # }
        #

        x_data_dict = dict()

        for header in self.header_list:
            x_data_dict[header] = dict()

            for i, data in enumerate(self.__get_raw_data(header)):

                # all of data convert to string
                if type(data) is int or type(data) is float:
                    data = str(data)

                data = data.strip()

                x_data_dict[header][i + POSITION_OF_ROW] = data

        self.x_data_count = len(x_data_dict[ID_COLUMN].values())

        return x_data_dict

    def __reset_data_dict(self, do_casting=False):
        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }

        def __dict2list(_do_casting=False):
            #
            # {
            #   column: { row: data }   --->   column: [ data_1, ..., data_n ]
            # }
            #

            if _do_casting:
                return [float(self.x_data_dict[header][index]) for index in self.x_data_dict[header]]
            else:
                return [self.x_data_dict[header][index] for index in self.x_data_dict[header]]

        for header in self.header_list:
            if do_casting and (self.get_type_of_column(header) == "scalar" or self.get_type_of_column(header) == "id"):
                self.x_data_dict[header] = __dict2list(do_casting)
            else:
                self.x_data_dict[header] = __dict2list()

        self.do_set_data = True

    def __summary(self):
        print("# of     all data set -", str(self.x_data_count).rjust(5),
              "\t# of mortality -", str(self.y_data_count).rjust(5))
        print("# of parsing data set -", str(self.x_data_count - len(self.erase_index_list)).rjust(5),
              "\t# of mortality -", str(self.counting_mortality(self.y_data)).rjust(5), "\n\n")

    def parsing(self):

        # set data
        self.__reset_data_dict()
        self.__summary()

    def __apply_exception(self):
        for header in self.header_list:
            for index in list(self.x_data_dict[header].keys()):
                if index in self.erase_index_list:
                    del self.x_data_dict[header][index]

    def __init_erase_index_list(self):

        # header keys 조건이 모두 만족 할 때
        def __condition(header_key, condition):
            _erase_index_dict = {i + POSITION_OF_ROW: 0 for i in range(self.x_data_count)}

            try:
                for index, value in self.x_data_dict[header_key].items():

                    if condition == 0:
                        if value == str(0) or value == str(0.0) or value == "nan":
                            _erase_index_dict[index] += 1
                    else:
                        if value == str(condition):
                            _erase_index_dict[index] += 1
            except KeyError:
                return dict()
            else:
                return _erase_index_dict

        def __append(_erase_index_dict):
            for index, _v in _erase_index_dict.items():
                if _v == 1 and index not in erase_index_list:
                    erase_index_list.append(index)

        def __case_of_exception_in_symptom(header_key="G"):
            for index, symptom in self.__get_raw_data(header_key).items():
                re_symptom = re.findall(r"[가-힣]+", symptom)
                if len(re_symptom) >= 1:
                    erase_index_list.append(index + POSITION_OF_ROW)

        def __set_column_target():
            if self.column_target:
                for index, symptom in self.__get_raw_data(self.column_target).items():
                    if not symptom == HAVE_SYMPTOM:
                        erase_index_list.append(index + POSITION_OF_ROW)

        erase_index_list = list()

        try:
            # column_initial_scalar 중 공백 혹은 -1의 데이터 제거
            for header in columns_dict["initial"]["scalar"]:
                __append(__condition(header_key=header, condition=float(0)))
                __append(__condition(header_key=header, condition=float(-1)))
        except KeyError:
            pass

        # erase which RR is 999 and TEMPERATURE is 99.9, 63.2
        __append(__condition(header_key=RR_COLUMN, condition=float(999)))
        __append(__condition(header_key=TEMP_COLUMN, condition=float(99.9)))
        __append(__condition(header_key=TEMP_COLUMN, condition=float(63.2)))

        # 피 검사 데이터가 많이 없는 경우
        for header in ["AJ", "AZ"]:
            __append(__condition(header_key=header, condition=float(0)))
            __append(__condition(header_key=header, condition="."))
            __append(__condition(header_key=header, condition="none"))

        # 주증상 데이터에 한글이 있는 경우의 예외처리
        __case_of_exception_in_symptom()

        # set target
        __set_column_target()

        # return list()
        return sorted(list(set(erase_index_list)), reverse=False)

    # DA is a column for y labels
    def __set_labels(self):
        y_labels = list()
        header = "DA"

        if self.do_parsing:
            for i, value in enumerate(self.__get_raw_data(header)):
                if i + POSITION_OF_ROW not in self.erase_index_list:
                    if value == HAVE_SYMPTOM:
                        y_labels.append([1])
                        self.y_data_count += 1
                    else:
                        y_labels.append([0])
                elif value == HAVE_SYMPTOM:
                    self.y_data_count += 1
        else:
            for i, value in enumerate(self.__get_raw_data(header)):
                if value == HAVE_SYMPTOM:
                    y_labels.append([1])
                else:
                    y_labels.append([0])

        return y_labels

    def get_type_of_column(self, column):
        for columns in self.columns_dict.values():
            for column_type, column_list in columns.items():
                if column in column_list:
                    return column_type

        return None

    def __free(self):
        if self.do_parsing:
            del self.__erase_index_list

        del self.__raw_data
        del self.__raw_header_dict
        del self.__x_data_count
        del self.__y_data_count
    
    @staticmethod
    def counting_mortality(data):
        count = 0
        for i in data:
            if i == [HAVE_SYMPTOM]:
                count += 1

        return count

    # show type of columns
    def show_type_of_columns(self):

        if not self.do_set_data:
            for header, data_lines in self.x_data_dict.items():
                type_dict = {"total": 0}

                for _, v in data_lines.items():
                    key = 0

                    if type(v) is float:
                        if math.isnan(v):
                            key = "float_nan"
                        else:
                            key = "float"
                    elif type(v) is str:
                        if v == "nan":
                            key = "nan"
                        else:
                            key = "str"
                    elif type(v) is int:
                        key = "int"

                    if key not in type_dict:
                        type_dict[key] = 1
                    else:
                        type_dict[key] += 1
                    type_dict["total"] += 1

                print(header.rjust(2), type_dict)
        else:
            for header, data_lines in self.x_data_dict.items():
                type_dict = {"total": 0}

                for v in data_lines:
                    key = 0
                    if type(v) is float:
                        if math.isnan(v):
                            key = "float_nan"
                        else:
                            key = "float"
                    elif type(v) is str:
                        if v == "nan":
                            key = "nan"
                        else:
                            key = "str"
                    elif type(v) is int:
                        key = "int"

                    if key not in type_dict:
                        type_dict[key] = 1
                    else:
                        type_dict[key] += 1
                    type_dict["total"] += 1

                print(header.rjust(2), type_dict)

        print("\n\n")

    def save(self, save_file_name):
        # copy raw data to raw data list except for erase index
        def __get_copied_raw_data(raw_data):
            _raw_data_list = list()

            for index in list(raw_data.keys()):
                if index + POSITION_OF_ROW not in self.erase_index_list:
                    _raw_data_list.append(raw_data[index])

            return _raw_data_list

        save_dict = OrderedDict()

        for header, header_key in self.raw_header_dict.items():
            if header in self.header_list:
                raw_data_list = self.x_data_dict[header]
            else:
                raw_data_list = __get_copied_raw_data(self.raw_data[header_key])

            random.seed(SEED)
            random.shuffle(raw_data_list)
            save_dict[header_key] = raw_data_list

        df = pd.DataFrame(save_dict)
        df.to_csv(DATA_PATH + save_file_name, index=False)

        print("Write csv file -", DATA_PATH + save_file_name, "\n")

    def load(self):
        self.__reset_data_dict(do_casting=True)
        self.__free()
