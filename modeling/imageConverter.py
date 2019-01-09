from DMP.learning.dataHandler import DataHandler
from DMP.utils.arg_convert_images import *
from PIL import Image
import numpy as np
import math
import os
import shutil


class ImageConverter:
    def __init__(self):
        self.dataHandler = DataHandler()
        self.dataHandler.show_info()

        # make directory
        if not os.path.isdir(DUMP_IMAGE):
            os.mkdir(DUMP_IMAGE)

        self.__make_dir(DUMP_IMAGE + SAVE_VECTOR + "/")
        self.__make_dir(DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TRAIN)
        self.__make_dir(DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_VALID)
        self.__make_dir(DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TEST)

    def convert(self):
        x_train = self.dataHandler.x_train
        x_valid = self.dataHandler.x_valid
        x_test = self.dataHandler.x_test

        # expand 1d to 2d which is a square matrix
        self.dataHandler.expand4square_matrix(*[x_train, x_valid, x_test])

        # save images
        self.__save_img(x_train, DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TRAIN)
        self.__save_img(x_valid, DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_VALID)
        self.__save_img(x_test, DUMP_IMAGE + SAVE_VECTOR + "/" + DUMP_TEST)

    @staticmethod
    def __save_img(x_data, _path):
        size = int(math.sqrt(len(x_data[0])))

        for i, data in enumerate(x_data):
            data = np.array(data)
            data = np.reshape(data, (-1, size))
            img = Image.fromarray(data)

            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(_path + str(i + 1) + '.jpg')

    @staticmethod
    def __make_dir(_path):
        if os.path.isdir(_path):
            shutil.rmtree(_path)

        os.mkdir(_path)
