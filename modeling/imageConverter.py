from DMP.learning.dataHandler import DataHandler
from DMP.utils.arg_convert_images import *
from PIL import Image
import numpy as np
import math
import os
import shutil
import json


class ImageConverter:
    def __init__(self):
        self.dataHandler = DataHandler()
        self.dataHandler.show_info()

        self.__log_path = LOG_PATH + LOG_NAME
        self.__train_path = DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TRAIN
        self.__valid_path = DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_VALID
        self.__test_path = DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TEST

        self.__make_dir(DUMP_IMAGE + SAVE_VECTOR + "/")
        self.__make_dir(self.train_path, have_to_make_labels=True)
        self.__make_dir(self.valid_path, have_to_make_labels=True)
        self.__make_dir(self.test_path, have_to_make_labels=True)

        self.__vector_dict = {
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
            'count_total_train': int(),
            'count_alive_train': int(),
            'count_death_train': int(),
            'count_total_valid': int(),
            'count_alive_valid': int(),
            'count_death_valid': int(),
            'count_total_test': int(),
            'count_alive_test': int(),
            'count_death_test': int()
        }

    @property
    def log_path(self):
        return self.__log_path

    @property
    def train_path(self):
        return self.__train_path

    @property
    def valid_path(self):
        return self.__valid_path

    @property
    def test_path(self):
        return self.__test_path

    @property
    def vector_dict(self):
        return self.__vector_dict

    @staticmethod
    def __make_dir(_path, have_to_make_labels=False):
        if os.path.isdir(_path):
            shutil.rmtree(_path)

        os.mkdir(_path)

        if have_to_make_labels:
            os.mkdir(_path + ALIVE_DIR)
            os.mkdir(_path + DEATH_DIR)

    def convert(self):
        x_train = self.dataHandler.x_train
        x_valid = self.dataHandler.x_valid
        x_test = self.dataHandler.x_test

        y_train = self.dataHandler.y_train
        y_valid = self.dataHandler.y_valid
        y_test = self.dataHandler.y_test

        # expand 1d to 2d which is a square matrix
        self.dataHandler.expand4square_matrix(*[x_train, x_valid, x_test])

        # save images
        self.__save_img(x_train, y_train, self.train_path, key="train")
        self.__save_img(x_valid, y_valid, self.valid_path, key="valid")
        self.__save_img(x_test, y_test, self.test_path, key="test")

    def __save_img(self, x_data, y_data, path, key):
        size = int(math.sqrt(len(x_data[0])))

        for i, data in enumerate(zip(x_data, y_data)):
            x, y = data[0], data[1]

            if y == [1]:
                save_path = path + DEATH_DIR
                self.vector_dict[key]["death"].append(i + 1)
                self.vector_dict['count_death_' + key] += 1
            else:
                save_path = path + ALIVE_DIR
                self.vector_dict[key]["alive"].append(i + 1)
                self.vector_dict['count_alive_' + key] += 1

            self.vector_dict['count_total_' + key] += 1

            x = np.array(x)
            x = np.reshape(x, (-1, size))
            img = Image.fromarray(x)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img.save(save_path + str(i + 1) + '.jpg')

    def save_log(self):
        with open(self.log_path, 'w') as outfile:
            json.dump(self.vector_dict, outfile, indent=4)
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", self.log_path, "\n\n")
