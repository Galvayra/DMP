import DMP.utils.arg_training as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .neuralNet import MyNeuralNetwork, MyPlot
import time


class DataClassifier:
    def __init__(self, data_handler):
        super().__init__()
        self.dataHandler = data_handler

    def training(self):
        start_time = time.time()

        x_train = self.dataHandler.x_train
        y_train = self.dataHandler.y_train
        x_valid = self.dataHandler.x_valid
        y_valid = self.dataHandler.y_valid
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        nn = MyNeuralNetwork()
        nn.init_plot()

        if op.MODEL_TYPE == "ffnn":
            nn.feed_forward_nn(x_train, y_train, x_test, y_test)
        elif op.MODEL_TYPE == "cnn":
            self.dataHandler.expand4square_matrix(*[x_train, x_valid, x_test])
            nn.convolution_nn(x_train, y_train, x_valid, y_valid, x_test, y_test)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")
        nn.show_plot()

    def predict(self):
        start_time = time.time()

        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        if op.MODEL_TYPE == "svm":
            x_train = self.dataHandler.x_train
            y_train = self.dataHandler.y_train
            svm = SVM()
            svm.init_plot()
            svm.svm(x_train, y_train, x_test, y_test)
            print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")
            svm.show_plot()
        elif op.MODEL_TYPE == "ffnn" or op.MODEL_TYPE == "cnn":
            nn = MyNeuralNetwork()
            nn.init_plot()
            nn.load_nn(x_test, y_test)
            print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")
            nn.show_plot()


class SVM(MyPlot):
    def __init__(self):
        super().__init__()

    def svm(self, x_train, y_train, x_test, y_test):
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
