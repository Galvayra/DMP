# -*- coding: utf-8 -*-
import pandas as pd
import math
import re
from .variables import *


# ### refer to reference file ###
class DataHandler:
    def __init__(self, data_file, is_reverse=False):
        # file name
        file_name = DATA_PATH + data_file
        print("Read csv file -", file_name, "\n\n")
        self.file_name = data_file

        if is_reverse:
            print("make reverse y labels!\n\n")

        self.__is_reverse = is_reverse
        
        # read csv file
        self.__raw_data = pd.read_csv(file_name)

        # header of data
        # [ 'C', 'E', .... 'CZ' ], E=4, CZ=103
        self.__header_list = self.__set_header_list(start=4, end=103)

        # a dictionary of header
        # { header: name of column }
        self.__head_dict = {self.__get_head_dict_key(i): v for i, v in enumerate(self.raw_data)}

        # a length of data
        self.__x_data_count = int()
        self.__y_data_count = int()

        # a dictionary of data
        # { header: a dictionary of data }
        self.x_data_dict = self.__init_x_data_dict()

        # a data of y labels
        # [ y_1, y_2, ... y_n ]
        self.y_data = list()

        # except for data which is not necessary
        # [ position 1, ... position n ]
        self.__erase_index_list = self.__init_erase_index_list()
        self.__apply_exception()

    @property
    def is_reverse(self):
        return self.__is_reverse

    @property
    def raw_data(self):
        return self.__raw_data

    @property
    def header_list(self):
        return self.__header_list

    @property
    def head_dict(self):
        return self.__head_dict

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

    def __set_header_list(self, start, end):
        # C is a identify number of patients
        header_list = [ID_COLUMN]
        header_list += [self.__get_head_dict_key(where) for where in range(start, end + 1)]

        return header_list

    def __init_x_data_dict(self):

        # {
        #   column: { row:data }
        #   C: { 2: C_1, 3: C_2, ... n: C_n }       ## ID
        #   E: { .... }                             ## Age
        #   ...
        #   CZ: { .... }                            ## Final Diagnosis
        # }
        #
        x_data_dict = dict()

        for header in self.header_list:
            header_key = self.head_dict[header]
            x_data_dict[header] = dict()

            for i, data in enumerate(self.raw_data[header_key]):

                # all of data convert to string
                if type(data) is int or type(data) is float:
                    data = str(data)

                data = data.strip()

                x_data_dict[header][i + POSITION_OF_ROW] = data

        self.x_data_count = len(x_data_dict[ID_COLUMN].values())

        return x_data_dict

    def __get_data_list(self, header):
        return [self.x_data_dict[header][index] for index in list(self.x_data_dict[header].keys())]

    def parsing(self):

        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }
        #

        for header in list(self.x_data_dict.keys()):
            self.x_data_dict[header] = self.__get_data_list(header)

            # print(header, len(self.x_data_dict[header]), type(self.x_data_dict[header]))

        self.y_data = self.__set_labels()

        print("# of     all data set =", str(self.x_data_count).rjust(5),
              "\t# of mortality =", self.y_data_count)
        print("# of parsing data set =", str(self.x_data_count - len(self.erase_index_list)).rjust(5),
              "\t# of mortality =", self.counting_mortality(self.y_data), "\n\n")

        self.free()

    def __apply_exception(self):
        for header in list(self.x_data_dict.keys()):
            for index in list(self.x_data_dict[header].keys()):
                if index in self.erase_index_list:
                    del self.x_data_dict[header][index]

    def __init_erase_index_list(self):

        # header keys 조건이 모두 만족 할 때
        def __condition(header_list, condition):
            # header_keys = [self.head_dict[i] for i in header_list]

            _erase_index_dict = {i + POSITION_OF_ROW: 0 for i in range(self.x_data_count)}

            for header_key in header_list:
                for index, value in self.x_data_dict[header_key].items():

                    if condition == 0:
                        if value == str(0) or value == str(0.0) or value == "nan":
                            _erase_index_dict[index] += 1
                    else:
                        if value == str(condition):
                            _erase_index_dict[index] += 1

            return _erase_index_dict, len(header_list)

        def __append(_erase_index_dict, _num_match, _individual=False):
            for index, _v in _erase_index_dict.items():
                if _individual and _v >= _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)
                elif not _individual and _v == _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)

        def __case_of_exception_in_symptom(header_key="G"):
            for index, symptom in self.raw_data[self.head_dict[header_key]].items():
                re_symptom = re.findall(r"[가-힣]+", symptom)
                if len(re_symptom) >= 1:
                    erase_index_list.append(index + POSITION_OF_ROW)

        # def __cut_random_data(_erase_index_list):
        #     r_num = int(CUT_RATIO.split('/')[1])
        #     cut_count = 0
        #     header_key = self.head_dict["DC"]
        #     for i, data in enumerate(self.raw_data[header_key]):
        #         if i not in _erase_index_list:
        #             if data != "사망":
        #                 cut_count += 1
        #                 if cut_count % r_num == 0:
        #                     _erase_index_list.append(i)

        erase_index_list = list()

        target_header_list = list()

        for v in columns_dict["initial"]["scalar"].values():
            for header in v:
                target_header_list.append(header)

        # column_initial_scalar 중 공백 혹은 -1의 데이터 제거
        for header in target_header_list:
            erase_index_dict, num_match = __condition(header_list=[header], condition=float(0))
            __append(erase_index_dict, num_match)

            erase_index_dict, num_match = __condition(header_list=[header], condition=float(-1))
            __append(erase_index_dict, num_match)

        # 피 검사 데이터가 많이 없는 경우
        for header in ["AJ", "AZ"]:
            erase_index_dict, num_match = __condition(header_list=[header], condition=float(0))
            __append(erase_index_dict, num_match)

            erase_index_dict, num_match = __condition(header_list=[header], condition=".")
            __append(erase_index_dict, num_match)

            erase_index_dict, num_match = __condition(header_list=[header], condition="none")
            __append(erase_index_dict, num_match)

        # 주증상 데이터에 한글이 있는 경우의 예외처리
        __case_of_exception_in_symptom()

        # print("num of", len(erase_index_list), "data is excepted!\n")

        return sorted(erase_index_list, reverse=False)

    # DC : 퇴원형태
    def __set_labels(self):
        y_labels = list()

        header_key = self.head_dict["DA"]

        if self.__is_reverse:
            for i, value in enumerate(self.raw_data[header_key]):
                if i + POSITION_OF_ROW not in self.erase_index_list:
                    if value == 1:
                        y_labels.append([0])
                    else:
                        y_labels.append([1])
                        self.y_data_count += 1
                elif value == 1:
                    self.y_data_count += 1
        else:
            for i, value in enumerate(self.raw_data[header_key]):
                if i + POSITION_OF_ROW not in self.erase_index_list:
                    if value == 1:
                        y_labels.append([1])
                        self.y_data_count += 1
                    else:
                        y_labels.append([0])
                elif value == 1:
                    self.y_data_count += 1

        return y_labels

    @staticmethod
    def get_type_of_column(column):
        for columns in columns_dict.values():
            for column_type, column_list in columns.items():
                if type(column_list) is dict:
                    for column_list_in_scalar in column_list.values():
                        if column in column_list_in_scalar:
                            return column_type

                if column in column_list:
                    return column_type

        return None

    def free(self):
        del self.__raw_data
        del self.__header_list
        del self.__head_dict
        del self.__erase_index_list
        del self.__x_data_count
        del self.__y_data_count
    
    @staticmethod
    def counting_mortality(data):
        count = 0
        for i in data:
            if i == [1]:
                count += 1

        return count

    def show_type_of_columns(self):
        # show result of columns inspecting
        for header, data_lines in self.x_data_dict.items():

            type_dict = {"total": 0}
            for _k, v in data_lines.items():

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
