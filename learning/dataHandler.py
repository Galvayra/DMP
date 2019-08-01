from .variables import GRAY_SCALE, INITIAL_IMAGE_SIZE
from PIL import Image
import numpy as np
import json
import math
import sys

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import READ_VECTOR, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET, IMAGE_PATH, VERSION, \
        show_options
elif current_script == "predict.py":
    from DMP.utils.arg_predict import READ_VECTOR, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET, IMAGE_PATH, VERSION, \
        show_options
elif current_script == "extract_feature.py" or current_script == "print_feature.py":
    from DMP.utils.arg_extract_feature import *
    from DMP.learning.plot import MyPlot
    from collections import OrderedDict
    from sklearn.ensemble import RandomForestClassifier
elif current_script == "convert_images.py":
    from DMP.utils.arg_convert_images import *
elif current_script == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import READ_VECTOR, DO_SHOW, VERSION, TYPE_OF_FEATURE, COLUMN_TARGET, show_options
    from DMP.modeling.variables import KEY_IMG_TEST, KEY_IMG_TRAIN, KEY_IMG_VALID
    from DMP.learning.variables import IMAGE_RESIZE, DO_NORMALIZE, DO_GRAYSCALE

alivePath = 'alive/'
deathPath = 'death/'
EXTENSION_OF_IMAGE = '.jpg'


class DataHandler:
    def __init__(self):
        try:
            with open(READ_VECTOR, 'r') as file:
                vector_list = json.load(file)
        except FileNotFoundError:
            print("\nPlease execute encoding script !")
            print("FileNotFoundError] READ_VECTOR is", "'" + READ_VECTOR + "'", "\n\n")
        else:
            print("\nRead vectors -", READ_VECTOR)

            show_options()

            # {
            #   feature: { 0: ["D", "header_name"], ... , n(dimensionality): ["CZ", "header_name"] }
            #   x_train: [ vector 1, ... vector n ], ... x_test, x_valid , ... , y_valid
            # }
            if current_script == "extract_feature.py":
                self.vector_matrix = OrderedDict()
                self.vector_matrix = {
                    "feature": dict(),
                    "x_train": dict(),
                    "y_train": list(),
                    "x_valid": dict(),
                    "y_valid": list(),
                    "x_test": dict(),
                    "y_test": list()
                }
                self.__importance = dict()

            self.feature = vector_list["feature"]
            self.x_train = vector_list["x_train"][TYPE_OF_FEATURE]
            self.y_train = vector_list["y_train"]
            self.x_valid = vector_list["x_valid"][TYPE_OF_FEATURE]
            self.y_valid = vector_list["y_valid"]
            self.x_test = vector_list["x_test"][TYPE_OF_FEATURE]
            self.y_test = vector_list["y_test"]

            if current_script == "fine_tuning.py":
                self.img_train = vector_list[KEY_IMG_TRAIN]
                self.img_valid = vector_list[KEY_IMG_VALID]
                self.img_test = vector_list[KEY_IMG_TEST]

            # count list
            # index == 0, all // index == 1, mortality // index == 2, alive
            self.count_all = list()
            self.count_mortality = list()
            self.count_alive = list()
            self.__set_count()

    @property
    def importance(self):
        return self.__importance

    def set_x_y_set(self, name_of_set="test"):
        if COLUMN_TARGET:
            if name_of_set == "train":
                x_target = self.x_train
                y_target = self.y_train
            elif name_of_set == "valid":
                x_target = self.x_valid
                y_target = self.y_valid
            else:
                x_target = self.x_test
                y_target = self.y_test

            target = list()
            target_index = list()

            for index_dim, feature in self.feature.items():
                if feature[0] == COLUMN_TARGET:
                    target.append(int(index_dim))

            for index, x in enumerate(x_target):
                if x[target[1]] == 1.0:
                    target_index.append(index)

            if name_of_set == "train":
                self.x_train = [x_target[index] for index in target_index]
                self.y_train = [y_target[index] for index in target_index]
            elif name_of_set == "valid":
                self.x_valid = [x_target[index] for index in target_index]
                self.y_valid = [y_target[index] for index in target_index]
            else:
                self.x_test = [x_target[index] for index in target_index]
                self.y_test = [y_target[index] for index in target_index]

    def __set_count(self):
        def __count_mortality(_y_data):
            _count = 0

            if len(_y_data[0]) > 1:
                death_vector = [0, 1]
            else:
                death_vector = [1]

            for _i in _y_data:
                if _i == death_vector:
                    _count += 1

            return _count

        self.count_all = [len(self.y_train), len(self.y_valid), len(self.y_test)]
        self.count_mortality = [__count_mortality(self.y_train),
                                __count_mortality(self.y_valid),
                                __count_mortality(self.y_test)]
        self.count_alive = [self.count_all[i] - self.count_mortality[i] for i in range(3)]

    def show_info(self):
        if DO_SHOW:
            print("\n\n\n======== DataSet Count ========")
            print("dims - ", len(self.x_train[0]))

            print("Training   Count -", str(self.count_all[0]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[0]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[0]).rjust(4))

            print("Validation Count -", str(self.count_all[1]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[1]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[1]).rjust(4))

            print("Test       Count -", str(self.count_all[2]).rjust(4),
                  "\t Mortality Count -", str(self.count_mortality[2]).rjust(3),
                  "\t Immortality Count -", str(self.count_alive[2]).rjust(4))

            print("\n\n======== DataSet Shape ========")
            x_train_np = np.array([np.array(j) for j in self.x_train])
            y_train_np = np.array([np.array(j) for j in self.y_train])
            print("Training   Set :", np.shape(x_train_np), np.shape(y_train_np))

            x_valid_np = np.array([np.array(j) for j in self.x_valid])
            y_valid_np = np.array([np.array(j) for j in self.y_valid])
            print("Validation Set :", np.shape(x_valid_np), np.shape(y_valid_np))

            y_test_np = np.array([np.array(j) for j in self.y_test])
            x_test_np = np.array([np.array(j) for j in self.x_test])
            print("Test       Set :", np.shape(x_test_np), np.shape(y_test_np), "\n\n")

    def set_image_path(self, vector_set_list, y_set_list, key_list):
        for key, vector_set, y_list in zip(key_list, vector_set_list, y_set_list):
            for enumerate_i, d in enumerate(zip(vector_set, y_list)):
                y_label = d[1]

                img_name = self.__get_name_of_image_from_index(key, enumerate_i, d[0])

                # img_name_list = [
                #     img_name + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_LR' + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_FLIP_LR_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_LR' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_TB' + EXTENSION_OF_IMAGE,
                #     img_name + '_ROTATE_FLIP_LR_TB' + EXTENSION_OF_IMAGE
                # ]

                if y_label == [1]:
                    vector_set[enumerate_i] = self.__get_image_from_path(IMAGE_PATH + deathPath + img_name)
                else:
                    vector_set[enumerate_i] = self.__get_image_from_path(IMAGE_PATH + alivePath + img_name)

    @staticmethod
    def __get_image_from_path(path, to_img=False):
        # normalize image of gray scale
        def __normalize_image(_img):
            gray_value = 255

            return np.array([[[k / gray_value for k in j] for j in i] for i in _img])

        def __change_gray_scale(_img):
            return np.array([[[j[0]] for j in i] for i in _img])

        img = Image.open(path)
        img.load()

        if IMAGE_RESIZE:
            img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE))

        new_img = np.asarray(img, dtype='int32')

        if to_img:
            if DO_NORMALIZE:
                new_img = __normalize_image(new_img)

            if DO_GRAYSCALE:
                new_img = __change_gray_scale(new_img)
        else:
            new_img = new_img.transpose([2, 0, 1]).reshape(3, -1)
            new_img = new_img[0]

            if DO_NORMALIZE:
                new_img = __normalize_image(new_img)

        return new_img

    def get_image_vector(self, x_data):
        """

        :param x_data:
        :return:
        [
            [img_1_vector_1, img_1_vector_2, ... , img_1_vector_i]
            [img_2_vector_1, img_2_vector_2, ... , img_2_vector_i]
            ...
            [img_N_vector_1, img_N_vector_2, ... , img_N_vector_i]
        ]
        """
        return [[self.__get_image_from_path(path, to_img=True) for path in paths[1]] for paths in x_data]

    def __get_name_of_image_from_index(self, key, enumerate_i, x_index):
        # k cross validation (version 1)
        if VERSION == 1:
            length_of_train = len(self.y_train)
            length_of_valid = length_of_train + len(self.y_valid)

            if x_index < length_of_train:
                return "train_" + str(x_index + 1) + EXTENSION_OF_IMAGE
            elif length_of_train <= x_index < length_of_valid:
                x_index -= length_of_train
                return "valid_" + str(x_index + 1) + EXTENSION_OF_IMAGE
            else:
                x_index -= length_of_valid
                return "test_" + str(x_index + 1) + EXTENSION_OF_IMAGE

        # optimize hyper-parameters (version 2)
        elif VERSION == 2:
            return key + "_" + str(enumerate_i + 1) + EXTENSION_OF_IMAGE
        else:
            return None

    @staticmethod
    def expand4square_matrix(*vector_set_list, use_origin=False):
        # origin data set       = [ [ v_1,      v_2, ... ,      v_d ],                       .... , [ ... ] ]
        # expand data set       = [ [ v_1,      v_2, ... ,      v_d,        0.0, ..., 0.0 ], .... , [ ... ] ]
        # gray scale data set   = [ [ v_1*255,  v_2*255, ... ,  v_d*255,    0.0, ..., 0.0 ], .... , [ ... ] ]
        size_of_1d = len(vector_set_list[0][0])

        if INITIAL_IMAGE_SIZE and not use_origin:
            size_of_2d = INITIAL_IMAGE_SIZE ** 2
        else:
            size_of_2d = pow(math.ceil(math.sqrt(size_of_1d)), 2)

        print("\n\nThe matrix size of vector - %d by %d" % (math.sqrt(size_of_2d), math.sqrt(size_of_2d)))

        for vector_set in vector_set_list:
            for i, vector in enumerate(vector_set):
                # expand data for 2d matrix
                for _ in range(size_of_1d, size_of_2d):
                    vector.append(0.0)

                vector_set[i] = [v * GRAY_SCALE for v in vector]

    def __random_forest(self):
        rf = RandomForestClassifier(n_estimators=NUM_OF_TREE, n_jobs=4, max_features='auto', random_state=0)
        return rf.fit(self.x_train + self.x_valid + self.x_test, self.y_train + self.y_valid + self.y_test)

    def __get_importance_features(self, feature, reverse=False):
        # reverse == T
        # --> get a not important features
        model = self.__random_forest()
        values = sorted(zip(feature.keys(), model.feature_importances_), key=lambda x: x[1] * -1)

        if reverse:
            return [(f[0], feature[f[0]], f[1]) for f in values if f[1] <= 0]
        else:
            return [(f[0], feature[f[0]], f[1]) for f in values if f[1] > 0]

    def show_importance_feature(self, reverse=False):
        feature_importance = self.__get_importance_features(self.feature, reverse=reverse)

        if reverse:
            if DO_SHOW:
                print("\n\nThere is not important feature")
                print("# of count -", len(feature_importance), "\n\n\n")
        else:
            for f in feature_importance:
                self.importance[f[1][0]] = [f[1][1], f[2]]

            if DO_SHOW:
                plot = MyPlot()
                plot.show_importance(feature_importance)

        if DO_SHOW:
            num_of_split_feature = 20
            for i, f in enumerate(feature_importance):
                print("%s (%s)\t %0.5f" % (str(f[1]).ljust(25), f[0], float(f[2])))
                if (i + 1) % num_of_split_feature == 0:
                        print("\n=======================================\n")

    def dump(self):
        with open(SAVE_LOG_NAME, 'w') as outfile:
            json.dump(self.importance, outfile, indent=4)
            print("\n=========================================================\n\n")
            print("success make dump file! - file name is", SAVE_LOG_NAME, "\n\n")
