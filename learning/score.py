from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from pandas import DataFrame
from .variables import *
from .plot import MyPlot
import copy
import sys
import time
from collections import OrderedDict

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW
elif sys.argv[0].split('/')[-1] == "predict.py":
    from DMP.utils.arg_predict import DO_SHOW, SAVE_DIR_NAME

NUM_OF_BLANK = 2
INDEX_OF_PERFORMANCE = 0


class MyScore(MyPlot):
    def __init__(self):
        super().__init__()
        self.__start_time = time.time()
        self.__score = self.__init_score()
        self.__score_dict = dict()
        self.__count_dict = dict()
        self.__num_of_fold = int()

    @property
    def start_time(self):
        return self.__start_time

    @property
    def score(self):
        return self.__score

    @property
    def count_dict(self):
        """
        {
          1 (k_fold):
            {
              'Training' :
               { "survive": int, "death": int, "total" : int},

              'Test' :
               { "survive": int, "death": int, "total" : int}
            } , ... ,

          K : { ... }
        }
        """
        return self.__count_dict

    @property
    def score_dict(self):
        """
        {
          1 (k_fold):
            {
              'alive' :
               { "Precision": float, "Recall": float, "F1 score": float, "Accuracy": float, "AUC" : float},

              'death' :
               { "Precision": float, "Recall": float, "F1 score": float, "Accuracy": float, "AUC" : float},

              'Total' :
               { "Precision": float, "Recall": float, "F1 score": float, "Accuracy": float, "AUC" : float}
            } , ... ,

          K : { ... }
        }
        """
        return self.__score_dict

    @property
    def num_of_fold(self):
        return self.__num_of_fold

    @num_of_fold.setter
    def num_of_fold(self, num_of_fold):
        self.__num_of_fold = num_of_fold

    @staticmethod
    def __init_score():
        score = {
            KEY_PRECISION: float(),
            KEY_RECALL: float(),
            KEY_F1: float(),
            KEY_ACCURACY: float(),
            KEY_AUC: float()
        }

        return score

    def __set_score(self, score):
        for measure, value in score.items():
            self.__score[measure] = value

    def set_score(self, target):
        if self.num_of_fold not in self.score_dict:
            self.score_dict[self.num_of_fold] = dict()

        if target not in self.score_dict[self.num_of_fold]:
            self.score_dict[self.num_of_fold][target] = dict()

        self.score_dict[self.num_of_fold][target] = copy.deepcopy(self.score)
        self.__score = self.__init_score()

    def compute_score(self, y, y_predict, hypothesis, accuracy=False):
        try:
            fpr, tpr, _ = roc_curve(y, hypothesis)
        except ValueError:
            print("\n\ncost is NaN !!")
            exit(-1)
        else:
            precision = precision_score(y, y_predict)
            recall = recall_score(y, y_predict)
            f1 = f1_score(y, y_predict)

            # # show how match y and prediction of y
            # print("\n y  y_predict")
            # for i, j in zip(y, y_predict):
            #     print(i, j)
            # print("\n\n")

            if not accuracy:
                accuracy = accuracy_score(y, y_predict)

            fpr *= 100
            tpr *= 100

            if DO_SHOW:
                if precision == 0 or recall == 0:
                    print("\n\n------------\nIt's not working")
                    print("Precision : %.1f, Recall : %.1f" % ((precision * 100), (recall * 100)))
                    print("\n------------")

            self.__set_score({
                KEY_PRECISION: (precision * 100),
                KEY_RECALL: (recall * 100),
                KEY_F1: (f1 * 100),
                KEY_ACCURACY: (accuracy * 100),
                KEY_AUC: (auc(fpr, tpr) / 100)
            })
            self.tpr = tpr
            self.fpr = fpr
            self.auc = (auc(fpr, tpr) / 100)

    def __show_score(self, target):
        if DO_SHOW:
            if target:
                print('\n\n======== Target is', target, "========\n")

            print('Precision : %.1f' % self.score_dict[self.num_of_fold][target][KEY_PRECISION])
            print('Recall    : %.1f' % self.score_dict[self.num_of_fold][target][KEY_RECALL])
            print('F1-Score  : %.1f' % self.score_dict[self.num_of_fold][target][KEY_F1])
            print('Accuracy  : %.1f' % self.score_dict[self.num_of_fold][target][KEY_ACCURACY])
            print('AUC       : %.1f' % self.score_dict[self.num_of_fold][target][KEY_AUC])

    def show_performance(self):
        if DO_SHOW:
            print("\n\n\n\n\n-------------------------------------------------------------------------------------")
            if self.num_of_fold == INDEX_OF_PERFORMANCE:
                print("\t\t\t\tSystem Performance")
            else:
                print("\t\t\t\t" + str(self.num_of_fold) + " Fold Performance\n")

                if self.count_dict:
                    print("Training  Count -", str(self.count_dict[self.num_of_fold][KEY_TRAIN][KEY_TOTAL]).rjust(4),
                          "\t Survive Count -",
                          str(self.count_dict[self.num_of_fold][KEY_TRAIN][KEY_IMMORTALITY]).rjust(3),
                          "\t Death   Count -",
                          str(self.count_dict[self.num_of_fold][KEY_TRAIN][KEY_MORTALITY]).rjust(4))

                    print("Test      Count -", str(self.count_dict[self.num_of_fold][KEY_TEST][KEY_TOTAL]).rjust(4),
                          "\t Survive Count -",
                          str(self.count_dict[self.num_of_fold][KEY_TEST][KEY_IMMORTALITY]).rjust(3),
                          "\t Death   Count -",
                          str(self.count_dict[self.num_of_fold][KEY_TEST][KEY_MORTALITY]).rjust(4), '\n')
            print("-------------------------------------------------------------------------------------")

        self.__show_score(target=KEY_IMMORTALITY)
        self.__show_score(target=KEY_MORTALITY)
        self.__show_score(target=KEY_TOTAL)

        if DO_SHOW:
            print("\n\n")

    def set_training_count(self, y_train, y_test):
        def __get_death_count(target_list):
            count = int()

            for x in target_list:
                if x == [1]:
                    count += 1

            return count

        if self.num_of_fold not in self.count_dict:
            self.count_dict[self.num_of_fold] = {
                KEY_TRAIN: {
                    KEY_IMMORTALITY: len(y_train) - __get_death_count(y_train),
                    KEY_MORTALITY: __get_death_count(y_train),
                    KEY_TOTAL: len(y_train)
                },
                KEY_TEST: {
                    KEY_IMMORTALITY: len(y_test) - __get_death_count(y_test),
                    KEY_MORTALITY: __get_death_count(y_test),
                    KEY_TOTAL: len(y_test)
                }
            }

    def save_score_cross_valid(self, best_epoch=None, num_of_dimension=None, num_of_hidden=None, learning_rate=None):
        save_name = PATH_RESULT + SAVE_DIR_NAME
        num_of_fold = len(self.score_dict)
        loop_cnt = (len(self.score) + NUM_OF_BLANK)

        data_frame = {
            "K fold": ["Total"] + ["" for _ in range(1, loop_cnt)],
            "Set": ["Training", "Validation", "Test"] + ["" for _ in range(3, loop_cnt)],
            "# of total": ["" for _ in range(loop_cnt)],
            "# of mortality": ["" for _ in range(loop_cnt)],
            "# of alive": ["" for _ in range(loop_cnt)],
            "": ["" for _ in range(loop_cnt * num_of_fold)],
            SAVE_DIR_NAME: [key for key in self.score] + ["" for _ in range(NUM_OF_BLANK)],
            KEY_IMMORTALITY: list(),
            KEY_MORTALITY: list(),
            KEY_TOTAL: list(),
            " ": ["" for _ in range(loop_cnt * num_of_fold)],
            "# of dimension": [num_of_dimension] + ["" for _ in range(1, loop_cnt * num_of_fold)],
            "Best Epoch": [best_epoch] + ["" for _ in range(1, loop_cnt * num_of_fold)],
            "# of hidden layer": [num_of_hidden] + ["" for _ in range(1, loop_cnt * num_of_fold)],
            "learning rate": [learning_rate] + ["" for _ in range(1, loop_cnt * num_of_fold)],
        }
        #
        # # set "K fold" column
        # for _k_fold in range(1, num_of_fold):
        #     data_frame["K fold"].append(str(_k_fold) + " fold")
        #
        #     for _ in range(loop_cnt - 1):
        #         data_frame["K fold"].append("")
        #
        # data_frame["Set"] *= num_of_fold
        # data_frame[SAVE_DIR_NAME] *= num_of_fold

        for k, v in data_frame.items():
            print(k, len(v), v)

        exit(-1)

    def save_score(self, data_handler, best_epoch=None, num_of_dimension=None, num_of_hidden=None, learning_rate=None):
        save_name = PATH_RESULT + SAVE_DIR_NAME
        loop_cnt = len(self.score)

        data_frame = {
            "Set": ["Training", "Validation", "Test"] + ["" for _ in range(3, loop_cnt)],
            "# of total": data_handler.count_all + ["" for _ in range(3, loop_cnt)],
            "# of mortality": data_handler.count_mortality + ["" for _ in range(3, loop_cnt)],
            "# of alive": data_handler.count_alive + ["" for _ in range(3, loop_cnt)],
            "": ["" for _ in range(loop_cnt)],
            SAVE_DIR_NAME: [key for key in self.score],
            KEY_IMMORTALITY: ["%0.2f" % s for s in self.score_dict[INDEX_OF_PERFORMANCE][KEY_IMMORTALITY].values()],
            KEY_MORTALITY: ["%0.2f" % s for s in self.score_dict[INDEX_OF_PERFORMANCE][KEY_MORTALITY].values()],
            KEY_TOTAL: ["%0.2f" % s for s in self.score_dict[INDEX_OF_PERFORMANCE][KEY_TOTAL].values()],
            " ": ["" for _ in range(loop_cnt)],
            "# of dimension": [num_of_dimension] + ["" for _ in range(1, loop_cnt)],
            "Best Epoch": [best_epoch] + ["" for _ in range(1, loop_cnt)],
            "# of hidden layer": [num_of_hidden] + ["" for _ in range(1, loop_cnt)],
            "learning rate": [learning_rate] + ["" for _ in range(1, loop_cnt)]
        }

        data_df = DataFrame(data_frame)
        data_df.to_csv(save_name + '.csv', index=False)

        if DO_SHOW:
            print("\n\ncomplete saving!! -", save_name, "\n")

    def set_2_class_score(self):
        self.set_score(target=KEY_TOTAL)

        for key in self.score:
            self.score[key] += (self.score_dict[self.num_of_fold][KEY_MORTALITY][key] +
                                self.score_dict[self.num_of_fold][KEY_IMMORTALITY][key]) / 2

        self.set_score(target=KEY_TOTAL)

    def set_performance(self):
        average_score = self.__get_average_score()
        self.__num_of_fold = INDEX_OF_PERFORMANCE

        for target in average_score:
            self.__score = average_score[target]
            self.set_score(target)

    def __get_average_score(self):
        average_score = {
            KEY_IMMORTALITY: self.__init_score(),
            KEY_MORTALITY: self.__init_score(),
            KEY_TOTAL: self.__init_score()
        }

        for k_fold in range(1, self.num_of_fold + 1):
            for target in average_score:
                for measure in self.score:
                    average_score[target][measure] += self.score_dict[k_fold][target][measure]

        for target in average_score:
            for measure in self.score:
                average_score[target][measure] /= self.num_of_fold

        return average_score

    def show_process_time(self):
        process_time = time.time() - self.start_time

        if DO_SHOW:
            print("\n\n processing time     --- %s seconds ---" % process_time, "\n\n")

        return process_time
