import DMP.utils.arg_training as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

        nn = MyNeuralNetwork()

        if op.MODEL_TYPE == "ffnn":
            nn.feed_forward_nn(x_train, y_train, x_valid, y_valid)
        elif op.MODEL_TYPE == "cnn":
            self.dataHandler.expand4square_matrix(*[x_train, x_valid])
            nn.convolution_nn(x_train, y_train, x_valid, y_valid)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")

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
        else:
            nn = MyNeuralNetwork()
            nn.init_plot()
            if op.MODEL_TYPE == "ffnn":
                nn.load_nn(x_test, y_test)
            elif op.MODEL_TYPE == "cnn":
                self.dataHandler.expand4square_matrix(*[x_test])
                nn.load_nn(x_test, y_test)
            print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")
            nn.show_plot()


class SVM(MyPlot):
    def __init__(self):
        super().__init__()

    def svm(self, x_train, y_train, x_test, y_test):
        model = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        test_probas_ = model.predict_proba(x_test)

        self.compute_score(y_test, y_predict, test_probas_[:, 1])
        self.set_score(target=KEY_MORTALITY)
        self.show_score(target=KEY_MORTALITY)
        self.set_plot(target=KEY_MORTALITY)
        self.show_plot()
