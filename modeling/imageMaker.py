import cv2
import numpy as np
from DMP.learning.variables import IMAGE_RESIZE, GRAY_SCALE
from .variables import EXTENSION_OF_IMAGE, EXTENSION_OF_PICKLE
import pickle
import gzip
from DMP.utils.progress_bar import show_progress_bar

DO_NORMALIZE = True


class ImageMaker:
    def __init__(self, save_path):
        self.__save_path = save_path
        self.count = 0

    @property
    def save_path(self):
        return self.__save_path

    def image2vector(self, img_path):
        img = self.__load_image(img_path)
        img_name = self.get_img_name_from_path(img_path)

        with gzip.open(self.save_path + img_name + EXTENSION_OF_PICKLE, 'wb') as f:
            pickle.dump(img, f)

    def get_matrix_from_pickle(self, x_img_paths, key="train"):
        img_matrix = list()
        i = 0
        total_len = 0

        for img_path_list in x_img_paths:
            total_len += len(img_path_list)

        for img_path_list in x_img_paths:
            for img_path in img_path_list:
                img_pickle_path = self.__get_pickle_path_from_path(img_path)
                img_vector = self.__get_vector_from_pickle(img_pickle_path)
                img_matrix.append(img_vector)
                show_progress_bar(i + 1, total=total_len, prefix="Load " + key + " Pickles")
                i += 1

        if DO_NORMALIZE:
            return np.array(img_matrix).astype(np.float32) / GRAY_SCALE
        else:
            return np.array(img_matrix).astype(np.float32)

    @staticmethod
    def __get_vector_from_pickle(img_path):
        with gzip.open(img_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except EOFError:

                print("EOFError] load image error !! -", img_path)
                exit(-1)

            return data

    @staticmethod
    def __load_image(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        return img

    @staticmethod
    def get_img_name_from_path(img_path):
        img_path = img_path.split('/')
        num_of_patient = img_path[-2]
        num_of_image = img_path[-1].split(EXTENSION_OF_IMAGE)[0]

        # return num_of_patient + "_" + num_of_image + EXTENSION_OF_PICKLE
        return num_of_patient + "_" + num_of_image

    def __get_pickle_path_from_path(self, img_path):
        return self.save_path + self.get_img_name_from_path(img_path) + EXTENSION_OF_PICKLE
