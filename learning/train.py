import numpy as np
import DMP.utils.arg_training as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .neuralNet import MyNeuralNetwork
import time
import math


class MyTrain(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def training(self):
        def __show_shape():
            def __count_mortality(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            if op.DO_SHOW:
                x_train_np = np.array([np.array(j) for j in x_train])
                y_test_np = np.array([np.array(j) for j in y_test])
                x_valid_np = np.array([np.array(j) for j in x_valid])
                y_valid_np = np.array([np.array(j) for j in y_valid])
                x_test_np = np.array([np.array(j) for j in x_test])
                y_train_np = np.array([np.array(j) for j in y_train])

                print("\n\n=====================================")
                print("\nDataSet Count")
                print("dims - ", len(x_train[0]))
                print("Training   Count -", len(y_train), "\t Mortality Count -", __count_mortality(y_train))
                print("Validation Count -", len(y_valid), "\t Mortality Count -", __count_mortality(y_valid))
                print("Test       Count -", len(y_test), "\t Mortality Count -", __count_mortality(y_test))

                print("\n\nDataSet Shape")
                print("Training   Set :", np.shape(x_train_np), np.shape(y_train_np))
                print("Validation Set :", np.shape(x_valid_np), np.shape(y_valid_np))
                print("Test       Set :", np.shape(x_test_np), np.shape(y_test_np))

        start_time = time.time()

        x_train = self.vector_list["x_train"]["merge"]
        y_train = self.vector_list["y_train"]
        x_valid = self.vector_list["x_valid"]["merge"]
        y_valid = self.vector_list["y_valid"]
        x_test = self.vector_list["x_test"]["merge"]
        y_test = self.vector_list["y_test"]

        __show_shape()

        self.init_plot()

        if op.MODEL_TYPE == "ffnn":
            self.feed_forward_nn(x_train, y_train, x_test, y_test)
        elif op.MODEL_TYPE == "cnn":
            self.expand4square_matrix(*[x_train, x_valid, x_test])
            self.convolution_nn(x_train, y_train, x_valid, y_valid, x_test, y_test)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")
        self.show_plot()

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

    def vector2txt(self, _file_name):
        def __vector2txt():
            def __write_vector(_w_file):
                for dimension, v in enumerate(x):
                    if v != 0:
                        _w_file.write(str(dimension + 1) + ":" + str(v) + token)
                _w_file.write("\n")

            with open("make/" + train_file_name + "_" + str(k_fold + 1) + ".txt", 'w') as train_file:
                for x, y in zip(x_train, y_train):
                    if y[0] == 1:
                        train_file.write(str(1) + token)
                    else:
                        train_file.write(str(-1) + token)
                    __write_vector(train_file)

            with open("make/" + test_file_name + "_" + str(k_fold + 1) + ".txt", 'w') as test_file:
                for x, y in zip(x_test, y_test):
                    if y[0] == 1:
                        test_file.write(str(1) + token)
                    else:
                        test_file.write(str(-1) + token)
                    __write_vector(test_file)

        token = " "
        train_file_name = "train_" + _file_name
        test_file_name = "test_" + _file_name

        # for k_fold in range(op.NUM_FOLDS):
        #     x_train = self.vector_list[k_fold]["x_train"]["merge"]
        #     x_test = self.vector_list[k_fold]["x_test"]["merge"]
        #     y_train = self.vector_list[k_fold]["y_train"]
        #     y_test = self.vector_list[k_fold]["y_test"]
        #
        #     __vector2txt()


class MyPredict(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def __svm(self, x_train, y_train, x_test, y_test):
        model = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
        model.fit(x_train, y_train)

        y_test_pred = model.predict(x_test)
        test_probas_ = model.predict_proba(x_test)

        _precision = precision_score(y_test, y_test_pred)
        _recall = recall_score(y_test, y_test_pred)
        _accuracy = accuracy_score(y_test, y_test_pred)
        _f1 = f1_score(y_test, y_test_pred)
        _svm_fpr, _svm_tpr, _ = roc_curve(y_test, test_probas_[:, 1])
        _svm_fpr *= 100
        _svm_tpr *= 100
        _auc = auc(_svm_fpr, _svm_tpr) / 100

        self.set_score(**{
            KEY_PRECISION: (_precision * 100),
            KEY_RECALL: (_recall * 100),
            KEY_F1: (_f1 * 100),
            KEY_ACCURACY: (_accuracy * 100),
            KEY_AUC: _auc
        })
        self.add_score(KEY_MORTALITY)
        self.show_score(target=KEY_MORTALITY, fpr=_svm_fpr, tpr=_svm_tpr)

    def predict(self):
        def __show_shape(_x_train=tuple(), _y_train=tuple()):
            def __count_mortality(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            x_test_np = np.array([np.array(j) for j in x_test])
            y_test_np = np.array([np.array(j) for j in y_test])

            print("\n\n=====================================")
            print("\nDataSet Count")
            print("dims - ", len(x_train[0]))
            if _x_train and _y_train:
                print("Training   Count -", len(_y_train), "\t Mortality Count -", __count_mortality(_y_train))
            print("Test       Count -", len(y_test), "\t Mortality Count -", __count_mortality(y_test))

            print("\n\nDataSet Shape")
            if _x_train and _y_train:
                x_train_np = np.array([np.array(j) for j in _x_train])
                y_train_np = np.array([np.array(j) for j in _y_train])
                print("Training   Set :", np.shape(x_train_np), np.shape(y_train_np))
            print("Test       Set :", np.shape(x_test_np), np.shape(y_test_np))

        self.init_plot()

        x_test = self.vector_list["x_test"]["merge"]
        y_test = self.vector_list["y_test"]

        if op.MODEL_TYPE == "svm":
            x_train = self.vector_list["x_train"]["merge"]
            y_train = self.vector_list["y_train"]

            if op.DO_SHOW:
                __show_shape(x_train, y_train)

            self.__svm(x_train, y_train, x_test, y_test)
        elif op.MODEL_TYPE == "ffnn" or op.MODEL_TYPE == "cnn":
            if op.DO_SHOW:
                __show_shape()

            self.load_nn(x_test, y_test)

        self.show_plot()

