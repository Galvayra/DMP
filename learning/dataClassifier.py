import sys
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from .variables import *
from .basicLearner import NeuralNet, ConvolutionNet
from .score import MyScore

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL, IMAGE_PATH, VERSION
elif current_script == "predict.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL, IMAGE_PATH, VERSION
elif current_script == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import TYPE_OF_MODEL, VERSION, DO_SHOW
    from DMP.learning.slimLearner import SlimLearner
    # from sklearn.preprocessing import StandardScaler
    from DMP.learning.variables import NUM_OF_K_FOLD
elif current_script == "predict_tfRecord.py":
    from DMP.utils.arg_predict_tfRecord import TYPE_OF_MODEL, DO_SHOW, VERSION
    from DMP.learning.slimLearner import SlimLearner

SEED = 1
DO_SHUFFLE = False
DO_IMG_SCALING = True
DO_SAMPLING = False
SAMPLE_RATIO = 5
IMG_D_TYPE = np.float32
n = 91


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
        nn = SlimLearner(model=TYPE_OF_MODEL, tf_name_vector=self.dataHandler.tf_name_vector)

        if not nn.is_cross_valid:
            nn.run_fine_tuning()
        else:
            pass
            # for _ in range(NUM_OF_K_FOLD):
            #     nn.run_fine_tuning()

        nn.save_process_time()

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

        x_data = x_train + x_valid + x_test
        y_data = y_train + y_valid + y_test

        return self.__get_set(x_data, y_data)

    @staticmethod
    def __get_set(x_data, y_data):
        if DO_SHUFFLE:
            print("\n==== Apply shuffle to the dataset ====\n\n")
            random.seed(SEED)
            c = list(zip(x_data, y_data))
            random.shuffle(c)
            x_data, y_data = zip(*c)

        return x_data, y_data

    @staticmethod
    def __get_data_matrix(_data, _index_list):
        return [_data[i] for i in _index_list]

    def __data_generator(self, x_data, y_data, cast_numpy=False, do_get_index=False):

        cv = KFold(n_splits=NUM_OF_K_FOLD, random_state=1, shuffle=True)

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

                if cast_numpy:
                    yield np.array(x_train, dtype=IMG_D_TYPE), np.array(y_train, dtype=IMG_D_TYPE), \
                          np.array(x_test, dtype=IMG_D_TYPE), np.array(y_test, dtype=IMG_D_TYPE)
                else:
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
                    ocf.predict(h, y_predict, y_test, is_cross_valid=True)

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
                        nn.predict(h, y_predict, y_test, is_cross_valid=True)
                elif TYPE_OF_MODEL == "cnn":
                    if IMAGE_PATH:
                        for _, y_train, x_test, y_test in self.__data_generator(x_data, y_data, do_get_index=True):
                            self.dataHandler.set_image_path([x_test], [y_test], key_list=["test"])
                            h, y_predict = nn.load_nn(x_test, y_test)
                            nn.set_training_count(y_train, y_test)
                            nn.predict(h, y_predict, y_test, is_cross_valid=True)
                    else:
                        for _, y_train, x_test, y_test in self.__data_generator(x_data, y_data):
                            self.dataHandler.expand4square_matrix(x_test)
                            h, y_predict = nn.load_nn(x_test, y_test)
                            nn.set_training_count(y_train, y_test)
                            nn.predict(h, y_predict, y_test, is_cross_valid=True)

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
                ocf.predict(h, y_predict, y_test, is_cross_valid=False)
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
                nn.predict(h, y_predict, y_test, is_cross_valid=False)
                nn.save(self.dataHandler)
                nn.show_process_time()
                nn.show_plot()

    def predict_tf_record(self):
        nn = SlimLearner(model=TYPE_OF_MODEL, tf_name_vector=self.dataHandler.tf_name_vector)

        if not nn.is_cross_valid:
            nn.load_nn()
            nn.save()

    # @staticmethod
    # def show_multi_plot():
    #     # initialize Neural Network
    #     nn = NeuralNet()
    #     nn.init_plot()
    #     nn.set_multi_plot()
    #     nn.show_plot()
    #
    # @staticmethod
    # def __img_scaling(x_train, x_test):
    #     if DO_IMG_SCALING:
    #         scaling = StandardScaler()
    #
    #         n, w, h, k = x_train.shape
    #         x_list = list(x_train.reshape([n, -1]))
    #
    #         scaling.fit(x_list)
    #         x_transformed = scaling.transform(x_list)
    #         x_train = np.array(x_transformed, dtype=IMG_D_TYPE).reshape([n, w, h, k])
    #
    #         n, w, h, k = x_test.shape
    #         x_list = list(x_test.reshape([n, -1]))
    #
    #         x_transformed = scaling.transform(x_list)
    #         x_test = np.array(x_transformed, dtype=IMG_D_TYPE).reshape([n, w, h, k])
    #
    #         return x_train, x_test
    #
    # def __get_total_image_set(self, x_data, y_data, has_img_paths=False):
    #     x_img_data = list()
    #     y_img_data = list()
    #
    #     if has_img_paths:
    #         if VERSION == 1:
    #             for images, y_value in zip(x_data[:n], y_data[:n]):
    #                 for img_path in images[1]:
    #                     x_img_data.append(img_path)
    #                     y_img_data.append(y_value)
    #         else:
    #             for images, y_value in zip(x_data[:n], y_data[:n]):
    #                 x_img_data.append(images[1])
    #                 y_img_data.append(y_value)
    #     else:
    #         if VERSION == 1:
    #             for images, y_value in zip(self.dataHandler.get_image_vector(x_data[:n]), y_data[:n]):
    #                 for image in images:
    #                     x_img_data.append(image)
    #                     y_img_data.append(y_value)
    #         else:
    #             for images, y_value in zip(self.dataHandler.get_image_vector(x_data[:n]), y_data[:n]):
    #                 x_img_data.append(images)
    #                 y_img_data.append(y_value)
    #
    #     self.__sampling(x_img_data, y_img_data)
    #     self.__show_info(y_img_data)
    #
    #     return self.__get_set(x_img_data, y_img_data)
    #
    # @staticmethod
    # def __sampling(x_img_data, y_img_data):
    #     if DO_SAMPLING:
    #         erase_index_list = list()
    #         cnt_of_death = int()
    #
    #         for i in range(len(y_img_data)):
    #             if y_img_data[i] == [0]:
    #                 erase_index_list.append(i)
    #             else:
    #                 cnt_of_death += 1
    #
    #         random.seed(5)
    #         random.shuffle(erase_index_list)
    #         random.seed(9)
    #         erase_index_list = random.sample(erase_index_list, len(erase_index_list) - (cnt_of_death * SAMPLE_RATIO))
    #
    #         for i in sorted(erase_index_list, reverse=True):
    #             del x_img_data[i], y_img_data[i]
    #
    # def __show_info_during_training(self, k_fold, y_train, y_test):
    #     print("\n\n============================", k_fold + 1, "- fold training ============================")
    #     self.__show_info(y_train, keyword="Training")
    #     self.__show_info(y_test, keyword="Test    ")
    #     print("\n\n\n\n")
    #
    # @staticmethod
    # def __show_info(y_img_data, keyword="Total Training"):
    #     if DO_SHOW:
    #         cnt = 0
    #
    #         if len(y_img_data[0]) > 1:
    #             for y in y_img_data:
    #                 if y == [0, 1]:
    #                     cnt += 1
    #         else:
    #             for y in y_img_data:
    #                 if y == [0]:
    #                     cnt += 1
    #
    #         print("\n" + keyword + " Count (alive/death) -", str(len(y_img_data)),
    #               '(' + str(cnt) + '/' + str(len(y_img_data) - cnt) + ')')


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

    def save(self, data_handler=False):
        self.set_performance()
        self.show_performance()

        if self.is_cross_valid:
            self.save_score_cross_valid()
        else:
            self.save_score(data_handler)
