# -*- coding: utf-8 -*-
from .variables import *
from DMP.dataset.dataHandler import DataHandler
import os
import shutil
import random


class ImageSplitter(DataHandler):
    def __init__(self):
        super().__init__(read_csv=READ_CSV, data_path=DATA_PATH, do_what="splitting")

        # initialize train test directory
        self.__init_train_test_dir()

        # set dictionary of ct images
        self.__ct_dict = dict()

        # set patient info from csv file
        self.__patient_number = [int(n) for n in self.x_data_dict[COLUMN_NUMBER].values()]
        self.__patient_hospital = self.x_data_dict[COLUMN_HOSPITAL].values()
        self.__patient_id = self.x_data_dict[COLUMN_ID].values()

        # alive, death list (consist of patient number) = ['1', '2', ... 'n']
        self.__alive_list = list()
        self.__death_list = list()
        self.__set_alive_death_list()

    @property
    def ct_dict(self):
        return self.__ct_dict

    @property
    def alive_list(self):
        return self.__alive_list

    @alive_list.setter
    def alive_list(self, alive_list):
        self.__alive_list = alive_list

    @property
    def death_list(self):
        return self.__death_list

    @property
    def patient_number(self):
        return self.__patient_number

    @property
    def patient_hospital(self):
        return self.__patient_hospital

    @staticmethod
    def __init_train_test_dir():
        def __make_dir__(_path):
            if os.path.isdir(_path):
                shutil.rmtree(_path)
            os.mkdir(_path)

            print("\nSuccess to make directory -", _path)

        __make_dir__(DATA_PATH + TRAIN_DIR)
        __make_dir__(DATA_PATH + TEST_DIR)

    def __show_count(self):
        print("\n# of     total -", str(len(self.alive_list) + len(self.death_list)).rjust(4),
              "\n# of     alive -", str(len(self.alive_list)).rjust(4),
              "\n# of mortality -", str(len(self.death_list)).rjust(4), "\n\n")

    def __set_alive_death_list(self):
        for n, y in zip(self.patient_number, self.y_data):
            if y == [0]:
                self.alive_list.append(n)
            else:
                self.death_list.append(n)

        self.__show_count()

        # sampling alive using death ratio
        self.alive_list = sorted(random.sample(self.alive_list, len(self.death_list)))

        print("After applying the sampling!")
        self.__show_count()

    def set_ct_dict(self):
        pass
