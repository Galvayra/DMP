import sys
from sklearn.svm import SVC
from .variables import *
from .neuralNet import MyNeuralNetwork
from .score import MyScore
import time

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL
elif current_script == "predict.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL


class DataClassifier:
    def __init__(self, data_handler):
        self.dataHandler = data_handler
        self.dataHandler.set_x_y_set(name_of_set="train")
        self.dataHandler.set_x_y_set(name_of_set="valid")
        self.dataHandler.set_x_y_set(name_of_set="test")
        self.dataHandler.show_info()

    def training(self):
        start_time = time.time()

        x_train = self.dataHandler.x_train
        y_train = self.dataHandler.y_train
        x_valid = self.dataHandler.x_valid
        y_valid = self.dataHandler.y_valid

        nn = MyNeuralNetwork()

        if TYPE_OF_MODEL == "ffnn":
            nn.feed_forward(x_train, y_train, x_valid, y_valid)
        elif TYPE_OF_MODEL == "cnn":
            self.dataHandler.expand4square_matrix(*[x_train, x_valid])
            nn.convolution(x_train, y_train, x_valid, y_valid)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")

    def predict(self):
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        if TYPE_OF_MODEL == "svm" or TYPE_OF_MODEL == "rf":
            x_train = self.dataHandler.x_train
            y_train = self.dataHandler.y_train

            ocf = OlderClassifier()
            ocf.init_plot()

            # initialize support vector machine
            h, y_predict = ocf.load_svm(x_train, y_train, x_test)
            ocf.predict(h, y_predict, y_test)
            ocf.save(self.dataHandler)
            ocf.show_plot()
        else:
            if TYPE_OF_MODEL == "cnn":
                self.dataHandler.expand4square_matrix(*[x_test])

            # initialize Neural Network
            nn = MyNeuralNetwork()
            nn.init_plot()
            h, y_predict = nn.load_nn(x_test, y_test)
            nn.predict(h, y_predict, y_test)
            nn.save(self.dataHandler)
            nn.show_plot()


class OlderClassifier(MyScore):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_svm(x_train, y_train, x_test):
        svc = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
        svc.fit(x_train, y_train)

        y_predict = svc.predict(x_test)
        test_probas_ = svc.predict_proba(x_test)

        return test_probas_[:, 1], y_predict

    def predict(self, h, y_predict, y_test):
        def __get_reverse(_y_labels, is_hypothesis=False):
            _y_labels_reverse = list()

            if is_hypothesis:
                for _y in _y_labels:
                    _y_labels_reverse.append([1 - _y])
            else:
                for _y in _y_labels:
                    if _y == [0]:
                        _y_labels_reverse.append([1])
                    else:
                        _y_labels_reverse.append([0])

            return _y_labels_reverse

        # set score of immortality
        self.compute_score(__get_reverse(y_predict), __get_reverse(y_test), __get_reverse(h, is_hypothesis=True))
        self.set_score(target=KEY_IMMORTALITY)
        self.show_score(target=KEY_IMMORTALITY)
        self.set_plot(target=KEY_IMMORTALITY)

        # set score of mortality
        self.compute_score(y_predict, y_test, h)
        self.set_score(target=KEY_MORTALITY)
        self.show_score(target=KEY_MORTALITY)
        self.set_plot(target=KEY_MORTALITY)

        # set total score of immortality and mortality
        self.set_total_score()
        self.show_score(target=KEY_TOTAL)

    def save(self, data_handler):
        self.save_score(data_handler)
