# -*- coding: utf-8 -*-
import pandas as pd
import re
import random
import sys
import json
from os import path, listdir
from .variables import *
from DMP.dataset.images.variables import *

if sys.argv[0].split('/')[-1] == "parsing.py":
    from DMP.utils.arg_parsing import SAVE_FILE_TOTAL, SAVE_FILE_TEST, SAVE_FILE_TRAIN, SAVE_FILE_VALID, RATIO, \
        EXTENSION_FILE, LOG_NAME, TEST_CSV_TARGET


HAVE_SYMPTOM = 1
PROPOSITION = 10
SEED = 1

# keys for log file
KEY_DOWN_SAMPLING = "down_sampling"
KEY_ERASE_INDEX = "erase_index"
KEY_SPLIT_RATIO = "split_ratio"


# ### refer to reference file ###
class DataHandler:
    def __init__(self, read_csv, data_path=path.dirname(path.abspath(__file__)) + '/', do_what="encoding",
                 do_sampling=False, column_target=False, eliminate_target=False, ct_image_path=""):

        self.__do_parsing = False
        self.__data_path = data_path

        # set path of csv file
        if do_what == "parsing":
            # execute parsing.py
            data_path += ORIGIN_PATH
            self.__do_parsing = True
        elif do_what == "encoding":
            # execute encoding.py
            data_path += PARSING_PATH

        # read csv file
        try:
            self.__raw_data = pd.read_csv(data_path + read_csv)
            print("Read csv file -", data_path + read_csv, "\n")
        except FileNotFoundError:
            print("FileNotFoundError]", data_path + read_csv, "\n")
            exit(-1)

        self.__column_target = column_target
        self.__columns_dict = columns_dict

        # # if use target option, eliminate target column in column dict because it is not necessary
        # if eliminate_target and column_target:
        #     for class_of_column in list(self.columns_dict.keys()):
        #         for type_of_column in list(self.columns_dict[class_of_column].keys()):
        #             if self.column_target in self.columns_dict[class_of_column][type_of_column]:
        #                 self.columns_dict[class_of_column][type_of_column].remove(self.column_target)

        # header of data using column_dict in variables.py
        # [ 'C', 'E', .... 'CZ' ]
        self.header_list = self.__set_header_list()

        # a dictionary of header in raw data (csv file)
        # { header: name of column }
        self.raw_header_dict = {self.__get_head_dict_key(i): v for i, v in enumerate(self.raw_data)}

        # a length of data
        self.__x_data_count = int()
        self.__y_data_count = int()

        # a dictionary of data
        # { header: a dictionary of data }
        self.x_data_dict = dict()
        self.__set_data_dict()

        # a data of y labels
        # [ y_1, y_2, ... y_n ]
        self.y_column = Y_COLUMN
        self.y_data = self.__set_labels()
        self.__do_sampling = do_sampling

        if self.do_parsing:
            self.__log_path = self.data_path + PARSING_PATH + SAVE_FILE_TOTAL.split(EXTENSION_FILE)[0]

            if LOG_NAME:
                log_path = self.data_path + PARSING_PATH + LOG_NAME

                try:
                    with open(log_path, 'r') as infile:
                        self.__log_dict = json.load(infile)
                except FileNotFoundError:
                    print("\nPlease make sure a path of log name -", log_path)
                    exit(-1)
                print("Read log file -", log_path, "\n")

                self.__erase_index_list = self.__get_items_in_log_dict(target_key=KEY_ERASE_INDEX)
            else:
                self.__log_dict = dict()

                if ct_image_path:
                    self.__ct_image_path = self.data_path + IMAGE_PATH + ct_image_path
                    self.__erase_index_list = self.__init_erase_index_list_for_ct_image()
                else:
                    self.__erase_index_list = self.__init_erase_index_list(data_path)
                    self.__set_erase_index_list()

            if self.column_target:
                self.__append_target_in_erase_index_list()

            self.__apply_exception()
            self.__save_dict = OrderedDict()
        else:
            self.__erase_index_list = list()

            if self.column_target:
                self.__append_target_in_erase_index_list()

            self.__apply_exception()

    @property
    def column_target(self):
        return self.__column_target

    @property
    def columns_dict(self):
        return self.__columns_dict

    @property
    def raw_test_data(self):
        return self.__raw_test_data

    @property
    def raw_data(self):
        return self.__raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self.__raw_data = raw_data

    @property
    def erase_index_list(self):
        return self.__erase_index_list

    @erase_index_list.setter
    def erase_index_list(self, erase_index_list):
        self.__erase_index_list = erase_index_list

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
    def do_sampling(self):
        return self.__do_sampling

    @property
    def data_path(self):
        return self.__data_path

    @property
    def ct_image_path(self):
        return self.__ct_image_path

    @property
    def save_dict(self):
        return self.__save_dict

    @property
    def log_path(self):
        return self.__log_path

    @property
    def log_dict(self):
        return self.__log_dict
    
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

    # if TEST_CSV_TARGET is inputted, find a duplicated rows with a origin csv file and a target csv file
    def __init_erase_index_list(self, data_path):
        if TEST_CSV_TARGET:
            try:
                self.__raw_test_data = pd.read_csv(data_path + TEST_CSV_TARGET)
                print("Read csv file -", data_path + TEST_CSV_TARGET, "\n")
            except FileNotFoundError:
                print("FileNotFoundError]", data_path + TEST_CSV_TARGET, "\n")
                exit(-1)

            index_list = list()

            for n_hospital_target, n_id_target, n_time_target in self.__generator_of_identity(self.raw_test_data):
                raw_index = POSITION_OF_ROW
                for n_hospital, n_id, n_time in self.__generator_of_identity(self.raw_data):
                    if n_hospital == n_hospital_target and n_id == n_id_target and n_time == n_time_target:
                        index_list.append(raw_index)
                    raw_index += 1

            return list(set(index_list))
        else:
            return list()

    def __generator_of_identity(self, target_df):
        hospital_data = list(target_df[self.raw_header_dict[COLUMN_HOSPITAL]])
        id_data = list(target_df[self.raw_header_dict[ID_COLUMN]])
        time_data = list(target_df[self.raw_header_dict[COLUMN_TIME]])

        for n_hospital, n_id, n_time in zip(hospital_data, id_data, time_data):
            yield n_hospital, n_id, n_time

    def __set_data_dict(self):

        # {
        #   column: { row: data }
        #   C: { 2: C_1, 3: C_2, ... n: C_n }       ## ID
        #   E: { .... }                             ## Age
        #   ...
        #   CZ: { .... }                            ## Final Diagnosis
        # }
        #
        def __append_data_dict(_header_list):
            for _header in _header_list:
                self.x_data_dict[_header] = dict()
                for _i, _data in enumerate(self.__get_raw_data(_header)):

                    # all of data convert to string
                    if type(_data) is int or type(_data) is float:
                        _data = str(_data)

                    _data = _data.strip()

                    self.x_data_dict[_header][_i + POSITION_OF_ROW] = _data

        __append_data_dict([COLUMN_NUMBER, COLUMN_HOSPITAL])
        __append_data_dict(self.header_list)
        self.x_data_count = len(self.x_data_dict[ID_COLUMN].values())

        return self.x_data_dict

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

    def __summary(self, down_sampling_count=0):
        print("\n\n# of     all data set -", str(self.x_data_count).rjust(5),
              "\t# of mortality -", str(self.y_data_count).rjust(5))
        print("# of parsing data set -", str(len(self.y_data) - down_sampling_count).rjust(5),
              "\t# of mortality -", str(self.counting_mortality(self.y_data)).rjust(5))
        print("\n=========================================================\n\n")

    def parsing(self):
        # set data
        self.__reset_data_dict()
        self.__set_save_dict()

    def __apply_exception(self):
        for header in self.header_list:
            for index in list(self.x_data_dict[header].keys()):
                if index in self.erase_index_list:
                    del self.x_data_dict[header][index]

        for index in sorted(self.erase_index_list, reverse=True):
            del self.y_data[index - POSITION_OF_ROW]

    def __set_erase_index_list(self):
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

        self.erase_index_list += erase_index_list
        self.erase_index_list = sorted(list(set(self.erase_index_list)), reverse=False)
        self.__add_to_log_dict(target=self.erase_index_list, target_key=KEY_ERASE_INDEX)

    def __init_erase_index_list_for_ct_image(self):
        erase_index_list = list()

        patient_list = [int(patient_number) for patient_number in listdir(self.ct_image_path + CT_IMAGE_ALL_PATH)]

        # except data(patient_number) which haven't ct image
        for index, num_id in enumerate(self.__get_raw_data(COLUMN_NUMBER)):
            if num_id not in patient_list:
                erase_index_list.append(index + POSITION_OF_ROW)

        self.__add_to_log_dict(target=erase_index_list, target_key=KEY_ERASE_INDEX)

        return erase_index_list

    # focus on target column
    def __append_target_in_erase_index_list(self):
        for index, symptom in self.__get_raw_data(self.column_target).items():
            if not symptom == HAVE_SYMPTOM:
                self.erase_index_list.append(index + POSITION_OF_ROW)

        self.erase_index_list = sorted(list(set(self.erase_index_list)), reverse=False)

    # Y_COLUMN(=DA) is a column for y labels
    def __set_labels(self):
        y_labels = list()

        for i, value in enumerate(self.__get_raw_data(self.y_column)):
            if value == HAVE_SYMPTOM:
                y_labels.append([1])
                self.y_data_count += 1
            else:
                y_labels.append([0])

        return y_labels

    def get_type_of_column(self, column):
        for columns in self.columns_dict.values():
            for type_of_column, column_of_list in columns.items():
                if column in column_of_list:
                    return type_of_column

        return None

    def get_class_of_column(self, column):
        for class_of_column, _ in self.columns_dict.items():
            for _, column_of_list in _.items():
                if column in column_of_list:
                    return class_of_column

        return None

    def __free(self):
        if self.do_parsing:
            del self.__erase_index_list

        del self.__raw_data
        del self.__x_data_count
        del self.__y_data_count
    
    @staticmethod
    def counting_mortality(data):
        count = 0
        for i in data:
            if i == [HAVE_SYMPTOM]:
                count += 1

        return count

    def save(self):
        train_dict, valid_dict, test_dict = self.__split_files()

        df_dict = {
            SAVE_FILE_TOTAL: self.save_dict,
            SAVE_FILE_TRAIN: train_dict,
            SAVE_FILE_VALID: valid_dict,
            SAVE_FILE_TEST: test_dict
        }

        for file_name, data_dict in df_dict.items():
            df = pd.DataFrame(data_dict)
            df.to_csv(self.data_path + PARSING_PATH + file_name, index=False)
            print("Write csv file -", self.data_path + PARSING_PATH + file_name)

            cnt_mortality = self.counting_mortality(data_dict[self.raw_header_dict[self.y_column]])
            cnt_total = len(data_dict[self.raw_header_dict[self.y_column]])
            print("\n# of     total -", str(cnt_total).rjust(4),
                  "\n# of     alive -", str(cnt_total - cnt_mortality).rjust(4),
                  "\n# of mortality -", str(cnt_mortality).rjust(4), "\n\n")

        with open(self.log_path, 'w') as outfile:
            json.dump(self.log_dict, outfile, indent=4)
            print("Write log file -", self.log_path, "\n\n")

    def save_log(self):
        """
        init dict of patient who have ct images
        key: patient_number
        value: alive(0) or death(1)
        """
        patient_dict = dict()
        save_path = self.data_path + IMAGE_PATH + IMAGE_LOG_PATH + IMAGE_LOG_NAME
        ct_path = self.data_path + IMAGE_PATH + CT_IMAGE_PATH + CT_IMAGE_TARGET_PATH
        target_folder_dict = dict()

        for folder_name in listdir(ct_path):
            p_number = folder_name.split('_')[0]
            target_folder_dict[str(p_number)] = folder_name

        if not path.isfile(save_path):
            for p_number, y in zip(self.save_dict[self.raw_header_dict[COLUMN_NUMBER]],
                                   self.save_dict[self.raw_header_dict[self.y_column]]):

                # if y == HAVE_SYMPTOM:
                #     y_value = 1
                # else:
                #     y_value = 0

                if str(p_number) in target_folder_dict:
                    patient_dict[str(p_number)] = target_folder_dict[str(p_number)]

            with open(save_path, 'w') as outfile:
                json.dump(patient_dict, outfile, indent=4)

    # copy raw data to raw data list except for erase index
    def __get_copied_raw_data(self, raw_data):
        _raw_data_list = list()

        for _index in list(raw_data.keys()):
            if _index + POSITION_OF_ROW not in self.erase_index_list:
                _raw_data_list.append(raw_data[_index])

        return _raw_data_list

    def __init_down_sampling_list(self, down_sampling_count):
        down_sampling_list = list()

        for _index, y in enumerate(self.y_data):
            if y == [0]:
                down_sampling_list.append(_index)

        down_sampling_list = sorted(random.sample(down_sampling_list, down_sampling_count), reverse=True)
        self.__add_to_log_dict(target=down_sampling_list, target_key=KEY_DOWN_SAMPLING)

        return down_sampling_list

    def __down_sampling(self, down_sampling_count):
        if LOG_NAME:
            down_sampling_list = self.__get_items_in_log_dict(target_key=KEY_DOWN_SAMPLING)
        else:
            down_sampling_list = self.__init_down_sampling_list(down_sampling_count)

        for header, header_key in self.raw_header_dict.items():
            if header in self.header_list:
                raw_data_list = self.x_data_dict[header]
            else:
                raw_data_list = self.__get_copied_raw_data(self.raw_data[header_key])

            for index in down_sampling_list:
                del raw_data_list[index]

            self.save_dict[header_key] = raw_data_list

    def __add_to_log_dict(self, target, target_key):
        if not LOG_NAME:
            self.log_dict[target_key] = target
    
    def __get_items_in_log_dict(self, target_key):
        return self.log_dict[target_key]

    def __set_save_dict(self):
        # set save dictionary for csv file
        # {
        #   header_1 : [ data_1 , ... , data_N ], ... , h
        #   header_I
        # }
        down_sampling_count = len(self.y_data) - (self.counting_mortality(self.y_data) * PROPOSITION)

        # apply down sampling
        if self.do_sampling and down_sampling_count > 0:
            self.__down_sampling(down_sampling_count)
            self.__summary(down_sampling_count)
        # do not apply sampling
        else:
            for header, header_key in self.raw_header_dict.items():
                if header in self.header_list:
                    raw_data_list = self.x_data_dict[header]
                else:
                    raw_data_list = self.__get_copied_raw_data(self.raw_data[header_key])

                self.save_dict[header_key] = raw_data_list

            self.__summary()

    # train:test:valid  --> 5 : 2.5 : 2.5
    # if ratio == 0.8   --> 8 :   1 :   1
    def __init_index_dict(self):
        index_dict = dict()
        split_ratio_dict = {
            "train": list(),
            "test": list(),
            "valid": list()
        }

        def __is_choice(ratio=0.5):
            if random.randrange(10) < (ratio * 10):
                return True
            else:
                return False

        index_list = [i for i in range(len(self.save_dict[self.raw_header_dict[ID_COLUMN]]))]
        # random.seed(SEED)
        random.shuffle(index_list)

        if LOG_NAME:
            split_ratio_dict = self.__get_items_in_log_dict(target_key=KEY_SPLIT_RATIO)

            for index in index_list:
                if index in split_ratio_dict["train"]:
                    index_dict[index] = "train"
                elif index in split_ratio_dict["test"]:
                    index_dict[index] = "test"
                else:
                    index_dict[index] = "valid"
        else:
            for index in index_list:
                if __is_choice(RATIO):
                    index_dict[index] = "train"
                    split_ratio_dict["train"].append(index)
                elif __is_choice():
                    index_dict[index] = "test"
                    split_ratio_dict["test"].append(index)
                else:
                    index_dict[index] = "valid"
                    split_ratio_dict["valid"].append(index)

        self.__add_to_log_dict(target=split_ratio_dict, target_key=KEY_SPLIT_RATIO)

        return index_dict

    def __split_files(self):
        def __copy(target, data_dict):
            for header in data_dict:
                target[header].append(data_dict[header][index])

        index_dict = self.__init_index_dict()
        train_dict = {header: list() for header in self.save_dict.keys()}
        valid_dict = {header: list() for header in self.save_dict.keys()}
        test_dict = {header: list() for header in self.save_dict.keys()}

        for index in range(len(index_dict)):
            which = index_dict[index]

            if which is "train":
                __copy(train_dict, self.save_dict)
            elif which is "valid":
                __copy(valid_dict, self.save_dict)
            elif which is "test":
                __copy(test_dict, self.save_dict)

        return train_dict, valid_dict, test_dict

    def load(self):
        self.__reset_data_dict(do_casting=True)
        self.__free()
