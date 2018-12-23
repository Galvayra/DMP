# -*- coding: utf-8 -*-
from .variables import *
from DMP.dataset.dataHandler import DataHandler
import os
import shutil
import random
import json


class ImageSplitter(DataHandler):
    def __init__(self):
        super().__init__(read_csv=READ_CSV, data_path=IMAGE_PATH + CSV_PATH, do_what="splitting")

        # initialize train test directory
        self.__train_path = IMAGE_PATH + CSV_PATH + TRAIN_DIR
        self.__test_path = IMAGE_PATH + CSV_PATH + TEST_DIR
        self.__init_train_test_dir()

        # set dictionary of ct images
        self.__ct_dict = {
            'train': {
                'alive': list(),
                'death': list()
            },
            'test': {
                'alive': list(),
                'death': list()
            },
            'count_total_train': int(),
            'count_alive_train': int(),
            'count_death_train': int(),
            'count_total_test': int(),
            'count_alive_test': int(),
            'count_death_test': int()
        }

        self.__nh_dict = dict()

        # set patient info from csv file
        self.__patient_number = [int(n) for n in self.x_data_dict[COLUMN_NUMBER].values()]
        self.__patient_hospital = self.x_data_dict[COLUMN_HOSPITAL].values()
        self.__patient_id = self.x_data_dict[COLUMN_ID].values()

        # alive, death list (consist of patient number) = ['1', '2', ... 'n']
        self.__alive_list = list()
        self.__death_list = list()
        self.__set_alive_death_list()

        # set images for training and test
        self.__set_ct_dict()

    @property
    def train_path(self):
        return self.__train_path

    @property
    def test_path(self):
        return self.__test_path

    @property
    def ct_dict(self):
        return self.__ct_dict

    @property
    def nh_dict(self):
        return self.__nh_dict

    @property
    def alive_list(self):
        return self.__alive_list

    @alive_list.setter
    def alive_list(self, alive_list):
        self.__alive_list = alive_list

    @property
    def death_list(self):
        return self.__death_list

    @death_list.setter
    def death_list(self, death_list):
        self.__death_list = death_list

    @property
    def patient_number(self):
        return self.__patient_number

    @property
    def patient_hospital(self):
        return self.__patient_hospital

    def __init_train_test_dir(self):
        def __make_dir__(_path):
            if os.path.isdir(_path):
                shutil.rmtree(_path)
            os.mkdir(_path)

            # print("\nSuccess to make directory -", _path)

        __make_dir__(self.train_path)
        __make_dir__(self.train_path + "/" + ALIVE_DIR)
        __make_dir__(self.train_path + "/" + DEATH_DIR)

        __make_dir__(self.test_path)
        __make_dir__(self.test_path + "/" + ALIVE_DIR)
        __make_dir__(self.test_path + "/" + DEATH_DIR)

    def __show_count(self):
        print("\n# of     total -", str(len(self.alive_list) + len(self.death_list)).rjust(4),
              "\t# of images -", self.__count_images(self.alive_list + self.death_list),
              "\n# of     alive -", str(len(self.alive_list)).rjust(4),
              "\t# of images -", self.__count_images(self.alive_list),
              "\n# of mortality -", str(len(self.death_list)).rjust(4),
              "\t# of images -", self.__count_images(self.death_list), "\n\n")

    def __set_alive_death_list(self):
        for h, n, y in zip(self.patient_hospital, self.patient_number, self.y_data):
            self.nh_dict[n] = h

            if y == [0]:
                self.alive_list.append(n)
            else:
                self.death_list.append(n)

        self.__show_count()

        # sampling alive using death ratio
        self.alive_list = sorted(random.sample(self.alive_list, len(self.death_list)))

        print("======== After applying the sampling! ========")
        self.__show_count()

    def copy_images(self):
        def __copy_images(_path_src, _path_dst, _files_src, _files_dst):
            for _file_src, _file_dst in zip(_files_src, _files_dst):
                shutil.copyfile(_path_src + _file_src, _path_dst + _file_dst)

        # copy images for training
        for i, n in enumerate(self.ct_dict["train"]["alive"] + self.ct_dict["train"]["death"]):
            h = self.nh_dict[n]
            path_src = IMAGE_PATH + h + "/" + str(n) + "/"

            if n in self.alive_list:
                path_dst = self.train_path + "/" + ALIVE_DIR
            else:
                path_dst = self.train_path + "/" + DEATH_DIR

            __copy_images(path_src, path_dst, self.__get_path_of_images(h, n), self.__get_new_name_of_image(h, n))

        # copy images for test
        for i, n in enumerate(self.ct_dict["test"]["alive"] + self.ct_dict["test"]["death"]):
            h = self.nh_dict[n]
            path_src = IMAGE_PATH + h + "/" + str(n) + "/"

            if n in self.alive_list:
                path_dst = self.test_path + "/" + ALIVE_DIR
            else:
                path_dst = self.test_path + "/" + DEATH_DIR

            __copy_images(path_src, path_dst, self.__get_path_of_images(h, n), self.__get_new_name_of_image(h, n))

    def __set_ct_dict(self):
        def __is_choice(ratio=0.5):
            if random.randrange(10) < (ratio * 10):
                return True
            else:
                return False

        # "train" : [1, 2, 3, 6, .... , n]
        # "test" : [4, 5, 7, 8, .... , m]

        for n in self.alive_list + self.death_list:
            if __is_choice(RATIO):
                self.ct_dict['count_total_test'] += 1

                if n in self.alive_list:
                    self.ct_dict['test']['alive'].append(n)
                    self.ct_dict['count_alive_test'] += 1
                else:
                    self.ct_dict['test']['death'].append(n)
                    self.ct_dict['count_death_test'] += 1
            else:
                self.ct_dict['count_total_train'] += 1

                if n in self.alive_list:
                    self.ct_dict['train']['alive'].append(n)
                    self.ct_dict['count_alive_train'] += 1
                else:
                    self.ct_dict['train']['death'].append(n)
                    self.ct_dict['count_death_train'] += 1

        # self.__sampling_ct_dict()

        print("======== After split train and test set! ========\n")
        print("# of train set -", str(self.ct_dict['count_total_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['train']['alive']) +
              self.__count_images(self.ct_dict['train']['death']))
        print("    # of alive -", str(self.ct_dict['count_alive_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['train']['alive']))
        print("    # of death -", str(self.ct_dict['count_death_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['train']['death']))
        print("\n# of test  set -", str(self.ct_dict['count_total_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['test']['alive']) +
              self.__count_images(self.ct_dict['test']['death']))
        print("    # of alive -", str(self.ct_dict['count_alive_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['test']['alive']))
        print("    # of death -", str(self.ct_dict['count_death_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['test']['death']), "\n")

    def __sampling_ct_dict(self, n=5):
        self.ct_dict["train"]["alive"] = sorted(random.sample(self.ct_dict["train"]["alive"], n))
        self.ct_dict["train"]["death"] = sorted(random.sample(self.ct_dict["train"]["death"], n))
        self.ct_dict["test"]["alive"] = sorted(random.sample(self.ct_dict["test"]["alive"], n))
        self.ct_dict["test"]["death"] = sorted(random.sample(self.ct_dict["test"]["death"], n))
        self.ct_dict["count_total_train"] = 2 * n
        self.ct_dict["count_alive_train"] = n
        self.ct_dict["count_death_train"] = n
        self.ct_dict["count_total_test"] = 2 * n
        self.ct_dict["count_alive_test"] = n
        self.ct_dict["count_death_test"] = n

    def save_ct_dict2log(self):
        log_file = IMAGE_PATH + NAME_LOG

        with open(log_file, 'w') as outfile:
            json.dump(self.ct_dict, outfile, indent=4)
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", log_file, "\n\n")

    def __count_images(self, number_list):
        count = 0

        for n in number_list:
            count += len(self.__get_path_of_images(self.nh_dict[n], n))

        return count

    @staticmethod
    def __get_path_of_images(h, n):
        try:
            return sorted(os.listdir(IMAGE_PATH + h + "/" + str(n) + "/"))
        except FileNotFoundError:
            print("FileNotFoundError: No such file or directory -", IMAGE_PATH + h + "/" + str(n) + "/")
            return list()

    def __get_new_name_of_image(self, h, n):
        return [h + "_" + str(n) + "_" + name for name in self.__get_path_of_images(h, n)]
