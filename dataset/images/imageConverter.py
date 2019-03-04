from DMP.learning.dataHandler import DataHandler
from DMP.utils.arg_convert_images import *
from PIL import Image
import numpy as np
import math
import os
import shutil
import json

EXTENSION_OF_IMAGE = '.jpg'


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
            if CONVERT_VERSION == 1:
                print("Convert Version 1\n")
                os.mkdir(_path + ALIVE_DIR)
                os.mkdir(_path + DEATH_DIR)
            elif CONVERT_VERSION == 2:
                print("Convert Version 2\n")
                os.mkdir(_path + DUMP_TRAIN)
                os.mkdir(_path + DUMP_TRAIN + ALIVE_DIR)
                os.mkdir(_path + DUMP_TRAIN + DEATH_DIR)
                os.mkdir(_path + DUMP_VALID)
                os.mkdir(_path + DUMP_VALID + ALIVE_DIR)
                os.mkdir(_path + DUMP_VALID + DEATH_DIR)
                os.mkdir(_path + DUMP_TEST)
                os.mkdir(_path + DUMP_TEST + ALIVE_DIR)
                os.mkdir(_path + DUMP_TEST + DEATH_DIR)
            else:
                print("Error] Convert version option!")
                print("       You must input 1 or 2\n\n")
                exit(-1)

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

        if CONVERT_VERSION == 1:
            self.__version_1(x_train, y_train, key="train")
            self.__version_1(x_valid, y_valid, key="valid")
            self.__version_1(x_test, y_test, key="test")
        elif CONVERT_VERSION == 2:
            self.__version_2(x_train, y_train, key="train")
            self.__version_2(x_valid, y_valid, key="valid")
            self.__version_2(x_test, y_test, key="test")

    @staticmethod
    def show_img_size(img_size):
        print("\n\nImage   size -", img_size)
        if IMAGE_SIZE:
            print("Image resize -", IMAGE_SIZE, "\n\n")

    def __version_1(self, x_data, y_data, key):
        """
        # convert version 0

            file name:
                train_1~N
                valid_1~M
                test_1~K

            arg.output/
                alive/
                    train_1.jpg
                    train_2.jpg
                    ....
                    valid_2.jpg
                    valid_4.jpg
                    ...
                    test_1.jpg
                    test_3.jpg
                death/
                    train_3.jpg
                    train_4.jpg
                    ....
                    valid_1.jpg
                    valid_3.jpg
                    ...
                    test_2.jpg
                    test_4.jpg
        """

        size = int(math.sqrt(len(x_data[0])))

        for i, data in enumerate(zip(x_data, y_data)):
            x, y = data[0], data[1]
            file_name = key + '_' + str(i + 1)

            if y == [1]:
                self.vector_dict[key]["death"].append(file_name + EXTENSION_OF_IMAGE)
                self.vector_dict['count_death_' + key] += 1
                save_path = self.save_path + DEATH_DIR
            else:
                self.vector_dict[key]["alive"].append(file_name + EXTENSION_OF_IMAGE)
                self.vector_dict['count_alive_' + key] += 1
                save_path = self.save_path + ALIVE_DIR

            self.vector_dict['count_total_' + key] += 1

            img = self.__init_img(x, size)
            self.__save_img(img, save_path, file_name)

            if DO_TRANSFORM:
                self.__save_img(img.rotate(90), save_path, file_name + '_ROTATE')

    def __version_2(self, x_data, y_data, key):
        """
        # convert version 2

            file name:
                train_1~N
                valid_1~M
                test_1~K

            arg.output/
                train/
                    alive/
                        train_1.jpg
                        train_2.jpg
                        train_3.jpg
                    death/
                        train_4.jpg
                valid/
                    alive/
                        valid_2.jpg
                        valid_3.jpg
                        valid_4.jpg
                    death/
                        valid_1.jpg
                test/
                    alive/
                        test_1.jpg
                        test_2.jpg
                        test_4.jpg
                    death/
                        test_3.jpg
        """
        size = int(math.sqrt(len(x_data[0])))

        for i, data in enumerate(zip(x_data, y_data)):
            x, y = data[0], data[1]
            file_name = key + '_' + str(i + 1)

            if y == [1]:
                self.vector_dict[key]["death"].append(file_name + EXTENSION_OF_IMAGE)
                self.vector_dict['count_death_' + key] += 1
                save_path = self.save_path + key + "/" + DEATH_DIR
            else:
                self.vector_dict[key]["alive"].append(file_name + EXTENSION_OF_IMAGE)
                self.vector_dict['count_alive_' + key] += 1
                save_path = self.save_path + key + "/" + ALIVE_DIR

            self.vector_dict['count_total_' + key] += 1

            img = self.__init_img(x, size)
            self.__save_img(img, save_path, file_name)

            if DO_TRANSFORM:
                self.__save_img(img.rotate(90), save_path, file_name + '_ROTATE')

    @staticmethod
    def __init_img(x, size):
        x = np.array(x)
        x = np.reshape(x, (-1, size))
        img = Image.fromarray(x)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if IMAGE_SIZE:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            
        return img
    
    @staticmethod
    def __save_img(img, save_path, file_name):
        img.save(save_path + file_name + EXTENSION_OF_IMAGE)

        if DO_TRANSFORM:
            img.transpose(Image.FLIP_LEFT_RIGHT).save(
                save_path + file_name + '_FLIP_LR' + EXTENSION_OF_IMAGE
            )

            img.transpose(Image.FLIP_TOP_BOTTOM).save(
                save_path + file_name + '_FLIP_TB' + EXTENSION_OF_IMAGE
            )

            img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(
                save_path + file_name + '_FLIP_LR_TB' + EXTENSION_OF_IMAGE
            )

    def save_log(self):
        with open(self.save_path + LOG_NAME, 'w') as outfile:
            json.dump(self.vector_dict, outfile, indent=4)
            print("\n=========================================================")
            print("\nsuccess make dump file! - file name is", self.save_path + LOG_NAME, "\n\n")

        # with open(self.log_path, 'w') as outfile:
        #     json.dump(self.vector_dict, outfile, indent=4)
        #     print("\n=========================================================")
        #     print("\nsuccess make dump file! - file name is", self.log_path, "\n\n")
