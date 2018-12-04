import sys
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .variables import *
from .neuralNet import MyNeuralNetwork
from .score import MyScore
import time

current_frame = sys.argv[0].split('/')[-1]

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL
else:
    from DMP.utils.arg_predict import TYPE_OF_MODEL


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

        if TYPE_OF_MODEL == "ffnn":
            nn.feed_forward_nn(x_train, y_train, x_valid, y_valid)
        elif TYPE_OF_MODEL == "cnn":
            self.dataHandler.expand4square_matrix(*[x_train, x_valid])
            nn.convolution_nn(x_train, y_train, x_valid, y_valid)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time), "\n\n")

    def predict(self):
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test
        feature = self.dataHandler.feature

        if TYPE_OF_MODEL == "svm" or TYPE_OF_MODEL == "rf":
            x_train = self.dataHandler.x_train
            y_train = self.dataHandler.y_train

            ocf = OlderClassifier()
            ocf.init_plot()

            # initialize support vector machine
            if TYPE_OF_MODEL == "svm":
                h, y_predict = ocf.load_svm(x_train, y_train, x_test)
            # initialize random forest
            else:
                h, y_predict = ocf.load_random_forest(x_train, y_train, x_test, feature)

            ocf.predict(h, y_predict, y_test)
            ocf.show_plot()
        else:
            if TYPE_OF_MODEL == "cnn":
                self.dataHandler.expand4square_matrix(*[x_test])

            # initialize Neural Network
            nn = MyNeuralNetwork()
            nn.init_plot()
            h, y_predict = nn.load_nn(x_test, y_test)
            nn.predict(h, y_predict, y_test)
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

    @staticmethod
    def load_random_forest(x_train, y_train, x_test, feature):
        rf = RandomForestClassifier(n_estimators=400, n_jobs=4)
        model = rf.fit(x_train, y_train)

        values = sorted(zip(feature.keys(), model.feature_importances_), key=lambda x: x[1] * -1)

        y_predict = rf.predict(x_test)
        test_probas_ = rf.predict_proba(x_test)

        feature_importance = [(feature[f[0]], f[1]) for f in values if f[1] > 0]
        # feature_importance = [(feature[f[0]], f[1]) for f in values if f[1] <= 0]

        for i, feature in enumerate(feature_importance):
            print(str(i + 1).rjust(3), str(feature[0]).ljust(25), feature[1])
            #
            # if i + 1 == 100:
            #     break

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

        # save score & show plot
        self.save_score()
        self.show_plot()
