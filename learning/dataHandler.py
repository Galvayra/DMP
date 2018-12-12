from .variables import GRAY_SCALE
import numpy as np
import json
import math
import sys

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import READ_VECTOR, show_options, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET
elif current_script == "predict.py":
    from DMP.utils.arg_predict import READ_VECTOR, show_options, DO_SHOW, TYPE_OF_FEATURE, COLUMN_TARGET
else:
    from DMP.utils.arg_extract_feature import *
    from collections import OrderedDict
    from sklearn.ensemble import RandomForestClassifier


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

            self.feature = vector_list["feature"]
            self.x_train = vector_list["x_train"][TYPE_OF_FEATURE]
            self.y_train = vector_list["y_train"]
            self.x_valid = vector_list["x_valid"][TYPE_OF_FEATURE]
            self.y_valid = vector_list["y_valid"]
            self.x_test = vector_list["x_test"][TYPE_OF_FEATURE]
            self.y_test = vector_list["y_test"]

            # count list
            # index == 0, all // index == 1, mortality // index == 2, alive
            self.count_all = list()
            self.count_mortality = list()
            self.count_alive = list()

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

    def show_info(self):
        def __count_mortality(_y_data_):
            _count = 0
            for _i in _y_data_:
                if _i == [1]:
                    _count += 1

            return _count

        self.count_all = [len(self.y_train), len(self.y_valid), len(self.y_test)]
        self.count_mortality = [__count_mortality(self.y_train),
                                __count_mortality(self.y_valid),
                                __count_mortality(self.y_test)]
        self.count_alive = [self.count_all[i] - self.count_mortality[i] for i in range(3)]

        if DO_SHOW:
            # print("\n\n\n======== DataSet Count ========")
            # print("dims - ", len(self.x_train[0]))
            #
            # print("Training   Count -", str(len(self.y_train)).rjust(4),
            #       "\t Mortality Count -", str(__count_mortality(self.y_train)).rjust(3),
            #       "\t Immortality Count -", str(len(self.y_train) - __count_mortality(self.y_train)).rjust(4))
            #
            # print("Validation Count -", str(len(self.y_valid)).rjust(4),
            #       "\t Mortality Count -", str(__count_mortality(self.y_valid)).rjust(3),
            #       "\t Immortality Count -", str(len(self.y_valid) - __count_mortality(self.y_valid)).rjust(4))
            #
            # print("Test       Count -", str(len(self.y_test)).rjust(4),
            #       "\t Mortality Count -", str(__count_mortality(self.y_test)).rjust(3),
            #       "\t Immortality Count -", str(len(self.y_test) - __count_mortality(self.y_test)).rjust(4))

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

    @staticmethod
    def expand4square_matrix(*vector_set_list):
        # origin data set       = [ [ v_1,      v_2, ... ,      v_d ],                       .... , [ ... ] ]
        # expand data set       = [ [ v_1,      v_2, ... ,      v_d,        0.0, ..., 0.0 ], .... , [ ... ] ]
        # gray scale data set   = [ [ v_1*255,  v_2*255, ... ,  v_d*255,    0.0, ..., 0.0 ], .... , [ ... ] ]
        for vector_set in vector_set_list:
            size_of_1d = len(vector_set[0])
            size_of_2d = pow(math.ceil(math.sqrt(size_of_1d)), 2)

            for i, vector in enumerate(vector_set):
                # expand data for 2d matrix
                for _ in range(size_of_1d, size_of_2d):
                    vector.append(0.0)

                vector_set[i] = [v * GRAY_SCALE for v in vector]

    # def vector2txt(self):
    #     def __write_vector(_w_file):
    #         for dimension, v in enumerate(x):
    #             if v != 0:
    #                 _w_file.write(str(dimension + 1) + ":" + str(v) + token)
    #         _w_file.write("\n")
    #
    #     token = " "
    #     train_file_name = "train_" + READ_VECTOR.split('/')[-1]
    #     valid_file_name = "test_" + READ_VECTOR.split('/')[-1]
    #     test_file_name = "test_" + READ_VECTOR.split('/')[-1]
    #
    #     for file_name, data in zip(
    #             (train_file_name, valid_file_name, test_file_name),
    #             ((self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test))):
    #         with open("make/" + file_name + ".txt", 'w') as train_file:
    #             for x, y in zip(data[0], data[1]):
    #                 if y[0] == 1:
    #                     train_file.write(str(1) + token)
    #                 else:
    #                     train_file.write(str(-1) + token)
    #             __write_vector(train_file)

    @staticmethod
    def get_importance_features(x_train, y_train, feature):
        rf = RandomForestClassifier(n_estimators=400, n_jobs=4)
        model = rf.fit(x_train, y_train)

        values = sorted(zip(feature.keys(), model.feature_importances_), key=lambda x: x[1] * -1)

        return [(f[0], feature[f[0]], f[1]) for f in values if f[1] > 0]

    def extract_feature(self):
        feature_importance = self.get_importance_features(self.x_train, self.y_train, self.feature)
        feature_importance_index = sorted([int(f[0]) for f in feature_importance], reverse=True)
        self.__set_vector_matrix(feature_importance_index, self.x_train, 'x_train', TYPE_OF_FEATURE)
        self.__set_vector_matrix(feature_importance_index, self.x_valid, 'x_valid', TYPE_OF_FEATURE)
        self.__set_vector_matrix(feature_importance_index, self.x_test, 'x_test', TYPE_OF_FEATURE)
        self.__set_vector_matrix(feature_importance_index, self.y_train, 'y_train')
        self.__set_vector_matrix(feature_importance_index, self.y_valid, 'y_valid')
        self.__set_vector_matrix(feature_importance_index, self.y_test, 'y_test')

        for new_key, key in enumerate(sorted(feature_importance_index)):
            self.vector_matrix['feature'][str(new_key)] = self.feature[str(key)]

    def __set_vector_matrix(self, feature_importance_index, target, _key, _type=False):
        if _type:
            self.vector_matrix[_key][_type] = list()

            for data in target:
                self.vector_matrix[_key][_type].append([data[i] for i in feature_importance_index])
        else:
            self.vector_matrix[_key] = [data for data in target]

    def dump(self):
        def __counting_mortality(_data):
            count = 0
            for _d in _data:
                if _d == [1]:
                    count += 1

            return count

        with open(SAVE_VECTOR, 'w') as outfile:
            json.dump(self.vector_matrix, outfile, indent=4)
            print("\n=========================================================\n\n")
            print("success make dump file! - file name is", SAVE_VECTOR)

        if DO_SHOW:
            print("\nTrain total count -", str(len(self.vector_matrix["x_train"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_train"])).rjust(4))
            print("Valid total count -", str(len(self.vector_matrix["x_valid"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_valid"])).rjust(4))
            print("Test  total count -", str(len(self.vector_matrix["x_test"]["merge"])).rjust(4),
                  "\tmortality count -", str(__counting_mortality(self.vector_matrix["y_test"])).rjust(4), "\n\n")