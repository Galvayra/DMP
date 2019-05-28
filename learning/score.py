from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from pandas import DataFrame
from .variables import *
from .plot import MyPlot
import copy
import sys

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW
elif sys.argv[0].split('/')[-1] == "predict.py":
    from DMP.utils.arg_predict import DO_SHOW, SAVE_DIR_NAME


class MyScore(MyPlot):
    def __init__(self):
        super().__init__()
        self.__score = self.__init_score()
        self.__score_dict = dict()
        self.__num_of_fold = int()

    @property
    def score(self):
        return self.__score

    @property
    def score_dict(self):
        return self.__score_dict

    @property
    def num_of_fold(self):
        return self.__num_of_fold

    @num_of_fold.setter
    def num_of_fold(self, num):
        self.__num_of_fold = num

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

    def set_score(self, target, k_fold):
        if k_fold not in self.score_dict:
            self.score_dict[k_fold] = dict()

        if target not in self.score_dict[k_fold]:
            self.score_dict[k_fold][target] = dict()

        self.score_dict[k_fold][target] = copy.deepcopy(self.score)
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

    def show_score(self, target, k_fold):
        if DO_SHOW:
            if target:
                print('\n\n\n\n======== Target is', target, "========\n")

            print('Precision : %.1f' % self.score_dict[k_fold][target][KEY_PRECISION])
            print('Recall    : %.1f' % self.score_dict[k_fold][target][KEY_RECALL])
            print('F1-Score  : %.1f' % self.score_dict[k_fold][target][KEY_F1])
            print('Accuracy  : %.1f' % self.score_dict[k_fold][target][KEY_ACCURACY])
            print('AUC       : %.1f' % self.score_dict[k_fold][target][KEY_AUC])

    def save_score(self, data_handler, is_cross_valid=True, best_epoch=None, num_of_dimension=None, num_of_hidden=None,
                   learning_rate=None):

        def __get_score_list(class_of_key):
            score_list = [float() for _ in range(loop_cnt)]

            for _k_fold in self.score_dict:
                for i, score in enumerate(self.score_dict[_k_fold][class_of_key].values()):
                    score_list[i] += score

            return score_list

        save_name = PATH_RESULT + SAVE_DIR_NAME
        loop_cnt = len(self.score_dict[1][KEY_TOTAL])

        data_frame = {
            "Set": ["Training", "Validation", "Test"] + ["" for _ in range(3, loop_cnt)],
            "# of total": data_handler.count_all + ["" for _ in range(3, loop_cnt)],
            "# of mortality": data_handler.count_mortality + ["" for _ in range(3, loop_cnt)],
            "# of alive": data_handler.count_alive + ["" for _ in range(3, loop_cnt)],
            "": ["" for _ in range(loop_cnt)],
            "# of dimension": [num_of_dimension] + ["" for _ in range(1, loop_cnt)],
            "Best Epoch": [best_epoch] + ["" for _ in range(1, loop_cnt)],
            "# of hidden layer": [num_of_hidden] + ["" for _ in range(1, loop_cnt)],
            "learning rate": [learning_rate] + ["" for _ in range(1, loop_cnt)],
            " ": ["" for _ in range(loop_cnt)],
            SAVE_DIR_NAME: [key for key in self.score],
            KEY_IMMORTALITY: ["%0.2f" % score for score in __get_score_list(KEY_IMMORTALITY)],
            KEY_MORTALITY: ["%0.2f" % score for score in __get_score_list(KEY_MORTALITY)],
            KEY_TOTAL: ["%0.2f" % score for score in __get_score_list(KEY_TOTAL)]
        }

        data_df = DataFrame(data_frame)
        data_df.to_csv(save_name + '.csv', index=False)

        if DO_SHOW:
            print("\n\ncomplete saving!! -", save_name, "\n")

    def set_2_class_score(self, k_fold):
        self.set_score(target=KEY_TOTAL, k_fold=self.num_of_fold)

        for key in self.score:
            self.score[key] += (self.score_dict[k_fold][KEY_MORTALITY][key] +
                                self.score_dict[k_fold][KEY_IMMORTALITY][key]) / 2

        self.set_score(target=KEY_TOTAL, k_fold=self.num_of_fold)
