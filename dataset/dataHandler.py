# -*- coding: utf-8 -*-
import pandas as pd
import math
from .variables import *


# ### refer to reference file ###
class DataHandler:
    def __init__(self, is_reverse=False):
        # file name
        file_name = DATA_PATH + DATA_FILE
        print("Read csv file -", file_name, "\n\n")
        self.file_name = DATA_FILE

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
        self.__data_count = int()

        # a dictionary of data
        # { header: a dictionary of data }
        self.x_data_dict = self.__init_x_data_dict()

        # a data of y labels
        # [ y_1, y_2, ... y_n ]
        self.y_data = list()

        # except for data which is not necessary
        # [ position 1, ... position n ]
        self.__erase_index_list = self.__init_erase_index_list()

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
    def data_count(self):
        return self.__data_count

    @data_count.setter
    def data_count(self, count):
        self.__data_count = count

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
                # parsing for raw data
                if type(data) is int:
                    data = float(data)
                elif type(data) is str:
                    data = data.strip()

                x_data_dict[header][i+POSITION_OF_ROW] = data

        self.data_count = len(x_data_dict[ID_COLUMN].values())

        return x_data_dict

    def parsing(self):

        # {
        #   column: [ data_1, ... , data_n ]
        #   C: [ C_1, C_2, ... , C_n ]          ## ID
        #   E: [ .... ]                         ## Age
        #   ...
        #   CZ: [ .... ]                        ## Final Diagnosis
        # }
        #

        def __init_data_list():
            data_list = list()

            for i, v in self.x_data_dict[header].items():
                data_list.append(v)

            return data_list

        for header in self.x_data_dict:
            self.x_data_dict[header] = __init_data_list()
            # print(header, len(self.x_data_dict[header]), type(self.x_data_dict[header]))

        self.y_data = self.__set_labels()
        self.free()

    def __init_erase_index_list(self):

        # header keys 조건이 모두 만족 할 때
        def __condition(header_list, condition):
            # header_keys = [self.head_dict[i] for i in header_list]

            _erase_index_dict = {i+POSITION_OF_ROW: 0 for i in range(self.data_count)}

            for header_key in header_list:
                for index, value in self.x_data_dict[header_key].items():
                    value = str(value)

                    if condition == 0:
                        if value == str(condition) or value == str(0.0) or value == "nan":
                            _erase_index_dict[index] += 1
                    else:
                        if value == str(condition):
                            _erase_index_dict[index] += 1

            return _erase_index_dict, len(header_list)

        def __append(_erase_index_dict, _num_match, _individual=False):
            for index, v in _erase_index_dict.items():
                if _individual and v >= _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)
                elif not _individual and v == _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)

        def __append_no_data(header_key="F"):
            for index, v in self.raw_data[self.head_dict[header_key]].items():
                if type(v) is float:
                    if math.isnan(v):
                        erase_index_list.append(index)
                else:
                    if v == "N/V":
                        erase_index_list.append(index)

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

        # erase_index_dict, num_match = __condition(header_list=["H"], condition=0)
        # __append(erase_index_dict, num_match)

        # H : 수축혈압, I : 이완혈압, J : 맥박수, K : 호흡수 == 0 제외
        # erase_index_dict, num_match = __condition(header_list=["H", "I", "J", "K"], condition=0)
        # __append(erase_index_dict, num_match)

        # erase_index_dict, num_match = __condition(header_list=["H", "I", "J", "K"], condition=-1)
        # __append(erase_index_dict, num_match)

        # 주증상 데이터가 없는 경우
        __append_no_data()

        # # 혈액관련 데이터가 없는 경우
        # erase_index_dict, num_match = __condition(header_list=["AE", "AF", "AG", "AH", "AI"], condition=0)
        # # erase_index_dict, num_match = __condition(header_list=["AE", "AF", "AG", "AH", "AI", "AM", "AN",
        # #                                                          "AO", "AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX",
        # #                                                          "AY", "BC", "BD", "BE", "BF", "BG", "BH", "BK", "BL"
        # #                                                          ], condition=0)
        # __append(erase_index_dict, 1, _individual=True)
        #
        # __cut_random_data(erase_index_list)
        #
        # print("num of", len(erase_index_list), "data is excepted!\n")

        return sorted(erase_index_list, reverse=True)

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
        else:
            for i, value in enumerate(self.raw_data[header_key]):
                if i + POSITION_OF_ROW not in self.erase_index_list:
                    if value == 1:
                        y_labels.append([1])
                    else:
                        y_labels.append([0])

        return y_labels

    def free(self):
        del self.__raw_data
        del self.__header_list
        del self.__head_dict
        del self.__erase_index_list
        del self.__data_count
    
    @staticmethod
    def counting_mortality(data):
        count = 0
        for i in data:
            if i == [1]:
                count += 1

        return count
