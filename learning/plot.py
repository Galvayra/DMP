from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from pandas import DataFrame
from .variables import *
import DMP.utils.arg_training as op
import matplotlib.pyplot as plt
import copy


class MyPlot:
    def __init__(self):
        self.__my_plot = None
        self.__tpr, self.__fpr = self.__init_plot()
        self.__score = self.__init_score()
        self.__score_dict = dict()

    @property
    def my_plot(self):
        return self.__my_plot

    @property
    def score(self):
        return self.__score

    @property
    def tpr(self):
        return self.__tpr

    @property
    def fpr(self):
        return self.__fpr
        
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
    
    @staticmethod
    def __init_plot():
        return None, None
        
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

            if op.DO_SHOW:
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
            self.__tpr = tpr
            self.__fpr = fpr

    def show_score(self, target):
        if op.DO_SHOW:
            if target:
                print('\n\n\n\n======== Target is', target, "========\n")

            print('Precision : %.1f' % self.score_dict[target][KEY_PRECISION])
            print('Recall    : %.1f' % self.score_dict[target][KEY_RECALL])
            print('F1-Score  : %.1f' % self.score_dict[target][KEY_F1])
            print('Accuracy  : %.1f' % self.score_dict[target][KEY_ACCURACY])
            print('AUC       : %.1f' % self.score_dict[target][KEY_AUC])

    def save_score(self):
        save_name = PATH_RESULT + op.SAVE_DIR_NAME

        data = {
            "": [key for key in self.score],
            KEY_IMMORTALITY: ["%0.2f" % score for score in self.score_dict[KEY_IMMORTALITY].values()],
            KEY_MORTALITY: ["%0.2f" % score for score in self.score_dict[KEY_MORTALITY].values()],
            KEY_TOTAL: ["%0.2f" % score for score in self.score_dict[KEY_TOTAL].values()]
        }

        # print([s for s in self.score])
        data_df = DataFrame(data)
        data_df.to_csv(save_name + '.csv', index=False)
        print("save complete -", save_name, "\n\n")

    def set_total_score(self):
        length = len(self.score_dict)

        for key in self.score_dict[KEY_MORTALITY]:
            self.score[key] = (self.score_dict[KEY_MORTALITY][key] + self.score_dict[KEY_IMMORTALITY][key]) / length
        self.set_score(target=KEY_TOTAL)

    def init_plot(self):
        if op.DO_SHOW_PLOT:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            self.__my_plot = plt.subplot2grid((2, 2), (0, 0))
            self.my_plot.set_ylabel("Sensitivity")
            self.my_plot.set_xlabel("100 - Specificity")

            if op.MODEL_TYPE == "svm":
                self.my_plot.set_title("Support Vector Machine")
            elif op.MODEL_TYPE == "ffnn":
                self.my_plot.set_title("Feed Forward Neural Network")
            elif op.MODEL_TYPE == "cnn":
                self.my_plot.set_title("Convolution Neural Network")

    def set_plot(self, target):
        if op.DO_SHOW_PLOT:
            # self.my_plot.plot(self.fpr, self.tpr, alpha=0.3,
            #                   label='%s AUC = %0.1f' % (target, self.score_dict[target][KEY_AUC]))
            self.my_plot.plot(self.fpr, self.tpr, alpha=0.3, label='AUC of %s' % target)
            self.__tpr, self.__fpr = self.__init_plot()

    def show_plot(self):
        if op.DO_SHOW_PLOT:
            self.my_plot.legend(loc="lower right")
            plt.show()
