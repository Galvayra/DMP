from .variables import GRAY_SCALE
import numpy as np
import json
import math
import sys

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import READ_VECTOR, show_options, DO_SHOW, TYPE_OF_FEATURE
else:
    from DMP.utils.arg_predict import READ_VECTOR, show_options, DO_SHOW, TYPE_OF_FEATURE


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

            self.feature = vector_list["feature"]
            self.x_train = vector_list["x_train"][TYPE_OF_FEATURE]
            self.y_train = vector_list["y_train"]
            self.x_valid = vector_list["x_valid"][TYPE_OF_FEATURE]
            self.y_valid = vector_list["y_valid"]
            self.x_test = vector_list["x_test"][TYPE_OF_FEATURE]
            self.y_test = vector_list["y_test"]

            self.show_info()

    def show_info(self):
        def __count_mortality(_y_data_):
            _count = 0
            for _i in _y_data_:
                if _i == [1]:
                    _count += 1

            return _count

        if DO_SHOW:
            print("\n\n\n======== DataSet Count ========")
            print("dims - ", len(self.x_train[0]))

            print("Training   Count -", str(len(self.y_train)).rjust(4),
                  "\t Mortality Count -", str(__count_mortality(self.y_train)).rjust(3),
                  "\t Immortality Count -", str(len(self.y_train) - __count_mortality(self.y_train)).rjust(4))

            print("Validation Count -", str(len(self.y_valid)).rjust(4),
                  "\t Mortality Count -", str(__count_mortality(self.y_valid)).rjust(3),
                  "\t Immortality Count -", str(len(self.y_valid) - __count_mortality(self.y_valid)).rjust(4))

            print("Test       Count -", str(len(self.y_test)).rjust(4),
                  "\t Mortality Count -", str(__count_mortality(self.y_test)).rjust(3),
                  "\t Immortality Count -", str(len(self.y_test) - __count_mortality(self.y_test)).rjust(4))

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

    def vector2txt(self):
        def __write_vector(_w_file):
            for dimension, v in enumerate(x):
                if v != 0:
                    _w_file.write(str(dimension + 1) + ":" + str(v) + token)
            _w_file.write("\n")

        token = " "
        train_file_name = "train_" + READ_VECTOR.split('/')[-1]
        valid_file_name = "test_" + READ_VECTOR.split('/')[-1]
        test_file_name = "test_" + READ_VECTOR.split('/')[-1]

        for file_name, data in zip(
                (train_file_name, valid_file_name, test_file_name),
                ((self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test))):
            with open("make/" + file_name + ".txt", 'w') as train_file:
                for x, y in zip(data[0], data[1]):
                    if y[0] == 1:
                        train_file.write(str(1) + token)
                    else:
                        train_file.write(str(-1) + token)
                __write_vector(train_file)
