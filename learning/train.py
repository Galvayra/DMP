import numpy as np
import DeepPredict.arguments as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .neuralNet import MyNeuralNetwork
import time


class MyTrain(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def __train_svm(self, k_fold, x_train, y_train, x_test, y_test):
        model = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        probas_ = model.predict_proba(x_test)

        _precision = precision_score(y_test, y_pred)
        _recall = recall_score(y_test, y_pred)
        _accuracy = accuracy_score(y_test, y_pred)
        _f1 = f1_score(y_test, y_pred)
        _svm_fpr, _svm_tpr, _ = roc_curve(y_test, probas_[:, 1])
        _svm_fpr *= 100
        _svm_tpr *= 100
        _auc = auc(_svm_fpr, _svm_tpr) / 100

        self.add_score(**{"P": _precision, "R": _recall, "F1": _f1, "Acc": _accuracy, "AUC": _auc})
        self.show_score(k_fold, fpr=_svm_fpr, tpr=_svm_tpr)

    def training(self):
        def __show_shape():
            def __count_mortality(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            x_train_np = np.array([np.array(j) for j in x_train])
            x_test_np = np.array([np.array(j) for j in x_test])
            y_train_np = np.array([np.array(j) for j in y_train])
            y_test_np = np.array([np.array(j) for j in y_test])

            print("\n\n\n\n=====================================\n")
            print("dims - ", len(x_train[0]))
            print("learning count -", len(y_train), "\t mortality count -", __count_mortality(y_train))
            print("test     count -", len(y_test), "\t mortality count -", __count_mortality(y_test), "\n")

            print(np.shape(x_train_np), np.shape(y_train_np))
            print(np.shape(x_test_np), np.shape(y_test_np))

        start_time = time.time()
        self.init_plot()

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            if op.DO_SHOW:
                __show_shape()

            if op.DO_SVM:
                self.__train_svm(k_fold, x_train, y_train, x_test, y_test)
            else:
                self.feed_forward_nn(k_fold, x_train, y_train, x_test, y_test)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time))
        print("\n\n")

        if op.DO_SVM:
            self.show_total_score("Support Vector Machine")
        else:
            self.show_total_score("Feed Forward NN")

        self.show_plot()

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

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            __vector2txt()


class MyPredict(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def predict(self):
        self.init_plot()

        for k_fold in range(op.NUM_FOLDS):
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_test = self.vector_list[k_fold]["y_test"]

            self.load_feed_forward_nn(k_fold, x_test, y_test)

        self.show_total_score("Feed Forward NN")
        self.show_plot()

