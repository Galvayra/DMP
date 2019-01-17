from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from pandas import DataFrame
from .variables import *
from .plot import MyPlot
import copy
import sys
#
# if sys.argv[0].split('/')[-1] == "training.py":
#     from DMP.utils.arg_training import DO_SHOW
# elif sys.argv[0].split('/')[-1] == "predict.py":
#     from DMP.utils.arg_predict import DO_SHOW, SAVE_DIR_NAME
#

DO_SHOW = True
SAVE_DIR_NAME = "save"


class MyScore(MyPlot):
    def __init__(self):
        super().__init__()
        self.__score = self.__init_score()
        self.__score_dict = dict()

    @property
    def score(self):
        return self.__score

    @property
    def score_dict(self):
        return self.__score_dict

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

    def __set_score(self, **score):
        for measure, value in score.items():
            self.__score[measure] = value

    def set_score(self, target):
        self.score_dict[target] = copy.deepcopy(self.score)
        self.__score = self.__init_score()

    def compute_score(self, y_predict, y, hypothesis, accuracy=False):
        try:
            fpr, tpr, _ = roc_curve(y, hypothesis)
        except ValueError:
            print("\n\ncost is NaN !!")
            exit(-1)
        else:
            precision = precision_score(y, y_predict)
            recall = recall_score(y, y_predict)
            f1 = f1_score(y, y_predict)

            if not accuracy:
                accuracy = accuracy_score(y, y_predict)

            fpr *= 100
            tpr *= 100

            if DO_SHOW:
                if precision == 0 or recall == 0:
                    print("\n\n------------\nIt's not working")
                    print("Precision : %.1f, Recall : %.1f" % ((precision * 100), (recall * 100)))
                    print("\n------------")

            self.__set_score(**{
                KEY_PRECISION: (precision * 100),
                KEY_RECALL: (recall * 100),
                KEY_F1: (f1 * 100),
                KEY_ACCURACY: (accuracy * 100),
                KEY_AUC: (auc(fpr, tpr) / 100)
            })
            self.tpr = tpr
            self.fpr = fpr

    def show_score(self, target):
        if DO_SHOW:
            if target:
                print('\n\n\n\n======== Target is', target, "========\n")

            print('Precision : %.1f' % self.score_dict[target][KEY_PRECISION])
            print('Recall    : %.1f' % self.score_dict[target][KEY_RECALL])
            print('F1-Score  : %.1f' % self.score_dict[target][KEY_F1])
            print('Accuracy  : %.1f' % self.score_dict[target][KEY_ACCURACY])
            print('AUC       : %.1f' % self.score_dict[target][KEY_AUC])

    def save_score(self, data_handler=None, best_epoch=None, num_of_dimension=None, num_of_hidden=None,
                   learning_rate=None):
        save_name = PATH_RESULT + SAVE_DIR_NAME

        data_frame = {
            "Set": ["Training", "Validation", "Test"] + ["" for _ in range(3, len(self.score_dict[KEY_TOTAL]))],
            "# of total": data_handler.count_all + ["" for _ in range(3, len(self.score_dict[KEY_TOTAL]))],
            "# of mortality": data_handler.count_mortality + ["" for _ in range(3, len(self.score_dict[KEY_TOTAL]))],
            "# of alive": data_handler.count_alive + ["" for _ in range(3, len(self.score_dict[KEY_TOTAL]))],
            "": ["" for _ in range(len(self.score_dict[KEY_TOTAL]))],
            "# of dimension": [num_of_dimension] + ["" for _ in range(1, len(self.score_dict[KEY_TOTAL]))],
            "Best Epoch": [best_epoch] + ["" for _ in range(1, len(self.score_dict[KEY_TOTAL]))],
            "# of hidden layer": [num_of_hidden] + ["" for _ in range(1, len(self.score_dict[KEY_TOTAL]))],
            "learning rate": [learning_rate] + ["" for _ in range(1, len(self.score_dict[KEY_TOTAL]))],
            " ": ["" for _ in range(len(self.score_dict[KEY_TOTAL]))],
            SAVE_DIR_NAME: [key for key in self.score],
            KEY_IMMORTALITY: ["%0.2f" % score for score in self.score_dict[KEY_IMMORTALITY].values()],
            KEY_MORTALITY: ["%0.2f" % score for score in self.score_dict[KEY_MORTALITY].values()],
            KEY_TOTAL: ["%0.2f" % score for score in self.score_dict[KEY_TOTAL].values()]
        }

        data_df = DataFrame(data_frame)
        data_df.to_csv(save_name + '.csv', index=False)

        if DO_SHOW:
            print("\n\ncomplete saving!! -", save_name, "\n")

    def set_total_score(self):
        length = len(self.score_dict)

        for key in self.score_dict[KEY_MORTALITY]:
            self.score[key] = (self.score_dict[KEY_MORTALITY][key] + self.score_dict[KEY_IMMORTALITY][key]) / length
        self.set_score(target=KEY_TOTAL)
