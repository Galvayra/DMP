import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from .variables import *
from .tensorLearner import NeuralNet, ConvolutionNet
from .score import MyScore
from PIL import Image

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL, IMAGE_PATH, VERSION
elif current_script == "predict.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL, IMAGE_PATH, VERSION
elif current_script == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import TYPE_OF_MODEL, VERSION, DO_SHOW
    from DMP.learning.transferLearner import TransferLearner


class DataClassifier:
    def __init__(self, data_handler=None):
        if data_handler:
            self.dataHandler = data_handler
            self.dataHandler.set_x_y_set(name_of_set="train")
            self.dataHandler.set_x_y_set(name_of_set="valid")
            self.dataHandler.set_x_y_set(name_of_set="test")

            if VERSION == 2:
                self.dataHandler.show_info()

    def training(self):
        nn = None

        if VERSION == 1:
            x_data, y_data = self.__get_total_set()

            if TYPE_OF_MODEL == "ffnn":
                nn = NeuralNet()
                for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                    nn.training(x_train, y_train, x_test, y_test)

            elif TYPE_OF_MODEL == "cnn":
                nn = ConvolutionNet()
                if IMAGE_PATH:
                    for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data, do_get_index=True):
                        self.dataHandler.set_image_path([x_train, x_test],
                                                        [y_train, y_test],
                                                        key_list=["train", "test"])
                        nn.training(x_train, y_train, x_test, y_test)
                else:
                    for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                        self.dataHandler.expand4square_matrix(x_train, x_test)
                        nn.training(x_train, y_train, x_test, y_test)

        elif VERSION == 2:
            x_train = self.dataHandler.x_train
            y_train = self.dataHandler.y_train
            x_valid = self.dataHandler.x_valid
            y_valid = self.dataHandler.y_valid

            if TYPE_OF_MODEL == "ffnn":
                nn = NeuralNet(is_cross_valid=False)
                nn.training(x_train, y_train, x_valid, y_valid)
            elif TYPE_OF_MODEL == "cnn":
                if IMAGE_PATH:
                    self.dataHandler.set_image_path([x_train, x_valid],
                                                    [y_train, y_valid],
                                                    key_list=["train", "valid"])
                else:
                    self.dataHandler.expand4square_matrix(x_train, x_valid)
                nn = ConvolutionNet(is_cross_valid=False)
                nn.training(x_train, y_train, x_valid, y_valid)

        nn.save_process_time()

    def transfer_learning(self):
        x_data, y_data = self.__get_total_set(has_img_paths=True)
        x_data, y_data = self.__get_total_image_set(x_data, y_data)

        if TYPE_OF_MODEL == "tuning":
            nn = TransferLearner()

            for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                nn.transfer_learning(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
                exit(-1)
        elif TYPE_OF_MODEL == "cnn":
            nn = ConvolutionNet()
            x_data = self.dataHandler.reshape_image_for_cnn(x_data)

            for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                nn.training(x_train, y_train, x_test, y_test, train_ct_image=True)

    def __get_total_set(self, has_img_paths=False):
        def __get_expended_x_data(vector_list, path_list):
            return [[vector] + [path] for vector, path in zip(vector_list, path_list)]

        if has_img_paths:
            x_train = __get_expended_x_data(self.dataHandler.x_train, self.dataHandler.img_train)
            x_valid = __get_expended_x_data(self.dataHandler.x_valid, self.dataHandler.img_valid)
            x_test = __get_expended_x_data(self.dataHandler.x_test, self.dataHandler.img_test)
        else:
            x_train = self.dataHandler.x_train
            x_valid = self.dataHandler.x_valid
            x_test = self.dataHandler.x_test

        y_train = self.dataHandler.y_train
        y_valid = self.dataHandler.y_valid
        y_test = self.dataHandler.y_test

        return x_train + x_valid + x_test, y_train + y_valid + y_test

    def __get_total_image_set(self, x_data, y_data):
        n = 3
        x_img_data = list()
        y_img_data = list()

        if VERSION == 1:
            for images, y_value in zip(self.dataHandler.get_image_vector(x_data[:n]), y_data[:n]):
                for image in images:
                    x_img_data.append(image)
                    y_img_data.append(y_value)
        else:
            for images, y_value in zip(self.dataHandler.get_image_vector(x_data[:n]), y_data[:n]):
                x_img_data.append(images)
                y_img_data.append(y_value)

        self.__show_info(y_img_data)

        return x_img_data, y_img_data

    @staticmethod
    def __show_info(y_img_data):
        if DO_SHOW:
            cnt = 0

            if len(y_img_data[0]) > 1:
                for y in y_img_data:
                    if y == [0, 1]:
                        cnt += 1
            else:
                for y in y_img_data:
                    if y == [0]:
                        cnt += 1

            print("\nTotal Training Count (alive/death) -", str(len(y_img_data)),
                  '(' + str(cnt) + '/' + str(len(y_img_data) - cnt) + ')', "\n\n")

    @staticmethod
    def __get_data_matrix(_data, _index_list):
        return [_data[i] for i in _index_list]

    def __data_generator(self, x_data, y_data, do_get_index=False):
        cv = KFold(n_splits=NUM_OF_K_FOLD, random_state=0, shuffle=False)

        if do_get_index:
            for train_index_list, test_index_list in cv.split(x_data, y_data):
                x_train = [int(i) for i in train_index_list]
                x_test = [int(i) for i in test_index_list]
                y_train = self.__get_data_matrix(y_data, train_index_list)
                y_test = self.__get_data_matrix(y_data, test_index_list)

                yield x_train, y_train, x_test, y_test
        else:
            for train_index_list, test_index_list in cv.split(x_data, y_data):
                x_train = self.__get_data_matrix(x_data, train_index_list)
                y_train = self.__get_data_matrix(y_data, train_index_list)
                x_test = self.__get_data_matrix(x_data, test_index_list)
                y_test = self.__get_data_matrix(y_data, test_index_list)

                yield x_train, y_train, x_test, y_test

    def predict(self):
        x_test = self.dataHandler.x_test
        y_test = self.dataHandler.y_test

        # k cross validation
        if VERSION == 1:
            x_data, y_data = self.__get_total_set()

            if TYPE_OF_MODEL == "svm":
                ocf = OlderClassifier()
                ocf.init_plot()

                for x_train, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                    h, y_predict = ocf.load_svm(x_train, y_train, x_test)
                    ocf.set_training_count(y_train, y_test)
                    ocf.predict(h, y_predict, y_test)

                ocf.save()
                ocf.show_process_time()
                ocf.show_plot()
            else:
                nn = NeuralNet()
                nn.init_plot()
                if TYPE_OF_MODEL == "ffnn":
                    for _, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                        h, y_predict = nn.load_nn(x_test, y_test)
                        nn.set_training_count(y_train, y_test)
                        nn.predict(h, y_predict, y_test)
                elif TYPE_OF_MODEL == "cnn":
                    if IMAGE_PATH:
                        for _, y_train, x_test, y_test in self.__data_generator(x_data, y_data, do_get_index=True):
                            self.dataHandler.set_image_path([x_test], [y_test], key_list=["test"])
                            h, y_predict = nn.load_nn(x_test, y_test)
                            nn.set_training_count(y_train, y_test)
                            nn.predict(h, y_predict, y_test)
                    else:
                        for _, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                            self.dataHandler.expand4square_matrix(x_test)
                            h, y_predict = nn.load_nn(x_test, y_test)
                            nn.set_training_count(y_train, y_test)
                            nn.predict(h, y_predict, y_test)

                nn.save()
                nn.show_process_time()
                nn.show_plot()

        elif VERSION == 2:
            if TYPE_OF_MODEL == "svm":
                x_train = self.dataHandler.x_train
                y_train = self.dataHandler.y_train

                ocf = OlderClassifier(is_cross_valid=False)
                ocf.init_plot()

                # initialize support vector machine
                h, y_predict = ocf.load_svm(x_train, y_train, x_test)
                ocf.predict(h, y_predict, y_test)
                ocf.save(self.dataHandler)
                ocf.show_process_time()
                ocf.show_plot()
            else:
                if TYPE_OF_MODEL == "cnn":
                    if IMAGE_PATH:
                        self.dataHandler.set_image_path([x_test], [y_test], key_list=["test"])
                    else:
                        self.dataHandler.expand4square_matrix(x_test)

                # initialize Neural Network
                nn = NeuralNet(is_cross_valid=False)
                nn.init_plot()
                h, y_predict = nn.load_nn(x_test, y_test)
                nn.predict(h, y_predict, y_test)
                nn.save(self.dataHandler)
                nn.show_process_time()
                nn.show_plot()

    @staticmethod
    def show_multi_plot():
        # initialize Neural Network
        nn = NeuralNet()
        nn.init_plot()
        nn.set_multi_plot()
        nn.show_plot()


class OlderClassifier(MyScore):
    def __init__(self, is_cross_valid=True):
        super().__init__()
        self.__is_cross_valid = is_cross_valid

    @property
    def is_cross_valid(self):
        return self.__is_cross_valid

    def load_svm(self, x_train, y_train, x_test):
        self.num_of_fold += 1
        svc = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
        svc.fit(x_train, self.get_y_set(y_train))

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
        self.compute_score(__get_reverse(self.get_y_set(y_test)),
                           __get_reverse(y_predict),
                           __get_reverse(h, is_hypothesis=True))
        self.set_score(target=KEY_IMMORTALITY)

        # set score of mortality
        self.compute_score(self.get_y_set(y_test), y_predict,  h)
        self.set_score(target=KEY_MORTALITY)

        # set total score of immortality and mortality
        self.set_2_class_score()

        if self.is_cross_valid:
            self.show_performance()

        self.set_plot(self.num_of_fold)

    def save(self, data_handler=False):
        self.set_performance()
        self.show_performance()

        if self.is_cross_valid:
            self.save_score_cross_valid()
        else:
            self.save_score(data_handler)
