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
        self.__save_path = DUMP_IMAGE + SAVE_VECTOR + "/"
        self.__make_dir(self.save_path, have_to_make_labels=True)

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
    def save_path(self):
        return self.__save_path

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
        self.dataHandler.expand4square_matrix(*[x_train, x_valid, x_test], use_origin=True)
        self.show_img_size(int(math.sqrt(len(x_train[0]))))

        # save images
        self.__save_img(x_train, y_train, key="train")
        self.__save_img(x_valid, y_valid, key="valid")
        self.__save_img(x_test, y_test, key="test")

    @staticmethod
    def show_img_size(img_size):
        print("\n\nImage   size -", img_size)
        if IMAGE_SIZE:
            print("Image resize -", IMAGE_SIZE, "\n\n")

    def __save_img(self, x_data, y_data, key):
        size = int(math.sqrt(len(x_data[0])))

        for i, data in enumerate(zip(x_data, y_data)):
            x, y = data[0], data[1]
            file_name = key + '_' + str(i + 1) + '.jpg'

            if y == [1]:
                self.vector_dict[key]["death"].append(file_name)
                self.vector_dict['count_death_' + key] += 1
                save_path = self.save_path + DEATH_DIR
            else:
                self.vector_dict[key]["alive"].append(file_name)
                self.vector_dict['count_alive_' + key] += 1
                save_path = self.save_path + ALIVE_DIR

            self.vector_dict['count_total_' + key] += 1

            x = np.array(x)
            x = np.reshape(x, (-1, size))
            img = Image.fromarray(x)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            if IMAGE_SIZE:
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img.save(save_path + file_name)

    def save_log(self):
        with open(self.log_path, 'w') as outfile:
            json.dump(self.vector_dict, outfile, indent=4)
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", self.log_path, "\n\n")
