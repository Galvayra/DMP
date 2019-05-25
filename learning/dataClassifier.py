import sys
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from .variables import *
from .neuralNet import MyNeuralNetwork
from .score import MyScore

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL, IMAGE_PATH, VERSION
elif current_script == "predict.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL, IMAGE_PATH, VERSION


class DataClassifier:
    def __init__(self, data_handler=None):
        if data_handler:
            self.dataHandler = data_handler
            self.dataHandler.set_x_y_set(name_of_set="train")
            self.dataHandler.set_x_y_set(name_of_set="valid")
            self.dataHandler.set_x_y_set(name_of_set="test")

            if VERSION == 1:
                self.dataHandler.show_info()

    def training(self):
        if VERSION == 1:
            nn = MyNeuralNetwork()
            x_data, y_data = self.__get_total_set()

            if TYPE_OF_MODEL == "ffnn":
                for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                    nn.feed_forward(x_train, y_train, x_test, y_test)
            elif TYPE_OF_MODEL == "cnn":
                # self.dataHandler.set_image_path method does not apply in cross validation!
                if IMAGE_PATH:
                    print("Do not use image path option !!")
                    print("You just input vectors!\n\n")
                    exit(-1)

                for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                    self.dataHandler.expand4square_matrix(x_train, x_test)
                    nn.convolution(x_train, y_train, x_test, y_test)

            nn.save_process_time()

        elif VERSION == 2:
            nn = MyNeuralNetwork(is_cross_valid=False)
            x_train = self.dataHandler.x_train
            y_train = self.dataHandler.y_train
            x_valid = self.dataHandler.x_valid
            y_valid = self.dataHandler.y_valid

            if TYPE_OF_MODEL == "ffnn":
                nn.feed_forward(x_train, y_train, x_valid, y_valid)
            elif TYPE_OF_MODEL == "cnn":
                if IMAGE_PATH:
                    self.dataHandler.set_image_path([x_train, x_valid], [y_train, y_valid], key_list=["train", "valid"])
                else:
                    self.dataHandler.expand4square_matrix(x_train, x_valid)
                nn.convolution(x_train, y_train, x_valid, y_valid)

            nn.save_process_time()

    def __get_total_set(self):
        x_train = self.dataHandler.x_train
        y_train = self.dataHandler.y_train
        x_valid = self.dataHandler.x_valid
        y_valid = self.dataHandler.y_valid
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        return x_train + x_valid + x_test, y_train + y_valid + y_test

    @staticmethod
    def __data_generator(x_data, y_data):
        def __get_data_matrix(_data, _index_list):
            return [_data[i] for i in _index_list]

        cv = KFold(n_splits=NUM_OF_K_FOLD, random_state=0, shuffle=False)

        for train_index_list, test_index_list in cv.split(x_data, y_data):
            x_train = __get_data_matrix(x_data, train_index_list)
            y_train = __get_data_matrix(y_data, train_index_list)
            x_test = __get_data_matrix(x_data, test_index_list)
            y_test = __get_data_matrix(y_data, test_index_list)

            yield x_train, y_train, x_test, y_test

    def __get_data_set(self, index_list):
        pass

    def predict(self):
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        if VERSION == 1:
            pass

        elif VERSION == 2:
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
                    if IMAGE_PATH:
                        self.dataHandler.set_image_path([x_test], [y_test], key_list=["test"])
                    else:
                        self.dataHandler.expand4square_matrix(x_test)

                # initialize Neural Network
                nn = MyNeuralNetwork()
                nn.init_plot()
                h, y_predict = nn.load_nn(x_test, y_test)
                nn.predict(h, y_predict, y_test)
                nn.save(self.dataHandler)
                nn.show_plot()
                nn.show_process_time()

    @staticmethod
    def show_multi_plot():
        # initialize Neural Network
        nn = MyNeuralNetwork()
        nn.init_plot()
        nn.set_multi_plot()
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

        self.num_of_fold += 1

        # set score of immortality
        self.compute_score(__get_reverse(y_predict), __get_reverse(y_test), __get_reverse(h, is_hypothesis=True))
        self.set_score(target=KEY_IMMORTALITY, k_fold=self.num_of_fold)
        self.show_score(target=KEY_IMMORTALITY, k_fold=self.num_of_fold)

        # set score of mortality
        self.compute_score(y_predict, y_test, h)
        self.set_score(target=KEY_MORTALITY, k_fold=self.num_of_fold)
        self.show_score(target=KEY_MORTALITY, k_fold=self.num_of_fold)
        self.set_plot()

        # set total score of immortality and mortality
        self.set_2_class_score()
        self.show_score(target=KEY_TOTAL, k_fold=self.num_of_fold)

    def save(self, data_handler):
        self.save_score(data_handler)
