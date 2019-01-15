# -*- coding: utf-8 -*-
from .variables import *
from DMP.dataset.dataHandler import DataHandler
import os
import shutil
import random
import json


class ImageSplitter(DataHandler):
    def __init__(self, has_ct_log=False):
        super().__init__(read_csv=READ_CSV, data_path=IMAGE_PATH + CSV_PATH, do_what="splitting")

        # log path
        self.__log_path = IMAGE_PATH + NAME_LOG

        # set patient info from csv file
        self.__patient_number = [int(n) for n in self.x_data_dict[COLUMN_NUMBER].values()]
        self.__patient_hospital = self.x_data_dict[COLUMN_HOSPITAL].values()
        self.__patient_id = self.x_data_dict[COLUMN_ID].values()

        # set patient number: patient hospital dict
        self.__nh_dict = dict()
        self.__set_nh_dict()

        # set dictionary of ct images
        if has_ct_log:
            # initialize train test directory
            self.__save_path = IMAGE_PATH + CSV_PATH + SAVE_DIR
            self.__init_directories()

            # load ct dictionary
            self.ct_dict = self.load_ct_dict()

        else:
            # alive, death list (consist of patient number) = ['1', '2', ... 'n']
            self.__alive_list = list()
            self.__death_list = list()

            # set alive and death list
            self.__set_alive_death_list()

            self.ct_dict = {
                'train': {
                    'alive': list(),
                    'death': list()
                },
                'valid': {
                    'alive': list(),
                    'death': list()
                },
                'test': {
                    'alive': list(),
                    'death': list()
                },
                'num_train': {
                    'alive': list(),
                    'death': list()
                },
                'num_valid': {
                    'alive': list(),
                    'death': list()
                },
                'num_test': {
                    'alive': list(),
                    'death': list()
                },
                'count_total_train': int(),
                'count_alive_train': int(),
                'count_death_train': int(),
                'count_total_valid': int(),
                'count_alive_valid': int(),
                'count_death_valid': int(),
                'count_total_test': int(),
                'count_alive_test': int(),
                'count_death_test': int(),
                'num_total_train': int(),
                'num_alive_train': int(),
                'num_death_train': int(),
                'num_total_valid': int(),
                'num_alive_valid': int(),
                'num_death_valid': int(),
                'num_total_test': int(),
                'num_alive_test': int(),
                'num_death_test': int()
            }

            self.__set_ct_dict()
            print("======== After split ========\n")

        self.__show_ct_dict()

    @property
    def save_path(self):
        return self.__save_path
    @property
    def log_path(self):
        return self.__log_path

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

    def __init_directories(self):
        def __make_dir__(_path):
            if os.path.isdir(_path):
                shutil.rmtree(_path)
            os.mkdir(_path)

        __make_dir__(self.save_path)
        __make_dir__(self.save_path + "/" + ALIVE_DIR)
        __make_dir__(self.save_path + "/" + DEATH_DIR)

    def __show_count(self):
        print("\n# of     total -", str(len(self.alive_list) + len(self.death_list)).rjust(4),
              "\t# of images -", self.__count_images(self.alive_list + self.death_list),
              "\n# of     alive -", str(len(self.alive_list)).rjust(4),
              "\t# of images -", self.__count_images(self.alive_list),
              "\n# of mortality -", str(len(self.death_list)).rjust(4),
              "\t# of images -", self.__count_images(self.death_list), "\n\n")

    def __set_nh_dict(self):
        for h, n in zip(self.patient_hospital, self.patient_number):
            self.nh_dict[n] = h

    def __set_alive_death_list(self):
        for n, y in zip(self.patient_number, self.y_data):

            if y == [0]:
                self.alive_list.append(n)
            else:
                self.death_list.append(n)

        self.__show_count()

        # sampling alive using death ratio
        self.alive_list = sorted(random.sample(self.alive_list, len(self.death_list)))

        print("======== After applying the sampling! ========")
        self.__show_count()

    def __load_alive_death_list(self):
        for value in self.ct_dict.values():
            if type(value) == dict:
                print(value)

    def __set_ct_dict(self):
        def __is_choice(ratio=0.5):
            if random.randrange(10) < (ratio * 10):
                return True
            else:
                return False

        # RATIO : (1-RATIO/2) : (1-RATIO/2)
        # if RATIO == 0.8
        #    train:valid:test = 8:1:1

        for n in self.alive_list + self.death_list:
            new_image_list = self.__get_new_name_of_image(self.nh_dict[n], n)

            if __is_choice(RATIO):
                self.ct_dict['num_total_train'] += 1
                self.ct_dict['count_total_train'] += len(new_image_list)

                if n in self.alive_list:
                    self.ct_dict['num_train']['alive'].append(n)
                    self.ct_dict['train']['alive'].extend(new_image_list)
                    self.ct_dict['num_alive_train'] += 1
                    self.ct_dict['count_alive_train'] += len(new_image_list)
                else:
                    self.ct_dict['num_train']['death'].append(n)
                    self.ct_dict['train']['death'].extend(new_image_list)
                    self.ct_dict['num_death_train'] += 1
                    self.ct_dict['count_death_train'] += len(new_image_list)
            elif __is_choice():
                self.ct_dict['num_total_valid'] += 1
                self.ct_dict['count_total_valid'] += len(new_image_list)

                if n in self.alive_list:
                    self.ct_dict['num_valid']['alive'].append(n)
                    self.ct_dict['valid']['alive'].extend(new_image_list)
                    self.ct_dict['num_alive_valid'] += 1
                    self.ct_dict['count_alive_valid'] += len(new_image_list)
                else:
                    self.ct_dict['num_valid']['death'].append(n)
                    self.ct_dict['valid']['death'].extend(new_image_list)
                    self.ct_dict['num_death_valid'] += 1
                    self.ct_dict['count_death_valid'] += len(new_image_list)
            else:
                self.ct_dict['num_total_test'] += 1
                self.ct_dict['count_total_test'] += len(new_image_list)

                if n in self.alive_list:
                    self.ct_dict['num_test']['alive'].append(n)
                    self.ct_dict['test']['alive'].extend(new_image_list)
                    self.ct_dict['num_alive_test'] += 1
                    self.ct_dict['count_alive_test'] += len(new_image_list)
                else:
                    self.ct_dict['num_test']['death'].append(n)
                    self.ct_dict['test']['death'].extend(new_image_list)
                    self.ct_dict['num_death_test'] += 1
                    self.ct_dict['count_death_test'] += len(new_image_list)

    def __show_ct_dict(self):
        print("# of total set -", str(self.ct_dict['num_total_train'] +
                                      self.ct_dict['num_total_valid'] +
                                      self.ct_dict['num_total_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['alive']) +
              self.__count_images(self.ct_dict['num_valid']['alive']) +
              self.__count_images(self.ct_dict['num_test']['alive']) +
              self.__count_images(self.ct_dict['num_train']['death']) +
              self.__count_images(self.ct_dict['num_valid']['death']) +
              self.__count_images(self.ct_dict['num_test']['death']))
        print("    # of alive -", str(self.ct_dict['num_alive_train'] +
                                      self.ct_dict['num_alive_valid'] +
                                      self.ct_dict['num_alive_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['alive']) +
              self.__count_images(self.ct_dict['num_valid']['alive']) +
              self.__count_images(self.ct_dict['num_test']['alive']))
        print("    # of death -", str(self.ct_dict['num_death_train'] +
                                      self.ct_dict['num_death_valid'] +
                                      self.ct_dict['num_death_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['death']) +
              self.__count_images(self.ct_dict['num_valid']['death']) +
              self.__count_images(self.ct_dict['num_test']['death']))

        print("\n\n# of train set -", str(self.ct_dict['num_total_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['alive']) +
              self.__count_images(self.ct_dict['num_train']['death']))
        print("    # of alive -", str(self.ct_dict['num_alive_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['alive']))
        print("    # of death -", str(self.ct_dict['num_death_train']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_train']['death']))

        print("\n# of valid set -", str(self.ct_dict['num_total_valid']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_valid']['alive']) +
              self.__count_images(self.ct_dict['num_valid']['death']))
        print("    # of alive -", str(self.ct_dict['num_alive_valid']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_valid']['alive']))
        print("    # of death -", str(self.ct_dict['num_death_valid']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_valid']['death']))

        print("\n# of test  set -", str(self.ct_dict['num_total_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_test']['alive']) +
              self.__count_images(self.ct_dict['num_test']['death']))
        print("    # of alive -", str(self.ct_dict['num_alive_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_test']['alive']))
        print("    # of death -", str(self.ct_dict['num_death_test']).rjust(3),
              '\t# of images -', self.__count_images(self.ct_dict['num_test']['death']), "\n")

    def save_ct_dict2log(self):
        with open(self.log_path, 'w') as outfile:
            json.dump(self.ct_dict, outfile, indent=4)
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", self.log_path, "\n\n")

    def load_ct_dict(self):
        with open(self.log_path, 'r') as infile:
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", self.log_path, "\n\n")
            return json.load(infile)

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

    def copy_images(self):
        def __copy_images(_path_src, _path_dst, _files_src, _files_dst):
            for _file_src, _file_dst in zip(_files_src, _files_dst):
                shutil.copyfile(_path_src + _file_src, _path_dst + _file_dst)

        # copy images for training
        for i, n in enumerate(self.ct_dict["num_train"]["alive"] + self.ct_dict["num_train"]["death"] +
                              self.ct_dict["num_valid"]["alive"] + self.ct_dict["num_valid"]["death"] +
                              self.ct_dict["num_test"]["alive"] + self.ct_dict["num_test"]["death"]):
            h = self.nh_dict[n]
            path_src = IMAGE_PATH + h + "/" + str(n) + "/"

            if n in (self.ct_dict["num_train"]["alive"] +
                     self.ct_dict["num_valid"]["alive"] +
                     self.ct_dict["num_test"]["alive"]):
                path_dst = self.save_path + "/" + ALIVE_DIR
            else:
                path_dst = self.save_path + "/" + DEATH_DIR

            __copy_images(path_src, path_dst, self.__get_path_of_images(h, n), self.__get_new_name_of_image(h, n))