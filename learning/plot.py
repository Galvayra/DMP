from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from .variables import KEY_PRECISION, KEY_RECALL, KEY_F1, KEY_ACCURACY, KEY_AUC
import matplotlib.pyplot as plt
import DMP.utils.arg_training as op
import copy


class MyPlot:
    def __init__(self):
        self.my_plot = None
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

    def set_score(self, **score):
        for measure, value in score.items():
            self.__score[measure] = value

    def add_score(self, target):
        self.score_dict[target] = copy.deepcopy(self.score)
        self.__init_score()

    def compute_score(self, y_test, h, p, acc):
        _precision = precision_score(y_test, p)
        _recall = recall_score(y_test, p)
        _f1 = f1_score(y_test, p)

        try:
            _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
        except ValueError:
            print("\n\ncost is NaN !!")
            exit(-1)
        else:
            _logistic_fpr *= 100
            _logistic_tpr *= 100
            _auc = auc(_logistic_fpr, _logistic_tpr) / 100

            if op.DO_SHOW:
                if _precision == 0 or _recall == 0:
                    print("\n\n------------\nIt's not working")
                    print("Precision : %.1f, Recall : %.1f" % ((_precision * 100), (_recall * 100)))
                    print("\n------------")

                self.my_plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3, label='AUC = %0.1f' % _auc)

            self.set_score(**{
                KEY_PRECISION: (_precision * 100),
                KEY_RECALL: (_recall * 100),
                KEY_F1: (_f1 * 100),
                KEY_ACCURACY: (acc * 100),
                KEY_AUC: _auc
            })

    def show_score(self, target, fpr, tpr):
        if op.DO_SHOW:
            if target:
                print('\n\nTarget    :', target)

            print('Precision : %.1f' % self.score_dict[target][KEY_PRECISION])
            print('Recall    : %.1f' % self.score_dict[target][KEY_RECALL])
            print('F1-Score  : %.1f' % self.score_dict[target][KEY_F1])
            print('Accuracy  : %.1f' % self.score_dict[target][KEY_ACCURACY])
            print('AUC       : %.1f' % self.score_dict[target][KEY_AUC])

            if fpr is not False and tpr is not False:
                self.my_plot.plot(fpr, tpr, alpha=0.3,
                                  label='%s AUC = %0.1f' % (target, self.score_dict[target][KEY_AUC]))

    def init_plot(self):
        if op.DO_SHOW:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            self.my_plot = plt.subplot2grid((2, 2), (0, 0))
            self.my_plot.set_ylabel("Sensitivity")
            self.my_plot.set_xlabel("100 - Specificity")

            if op.MODEL_TYPE == "svm":
                self.my_plot.set_title("Support Vector Machine")
            elif op.MODEL_TYPE == "ffnn":
                self.my_plot.set_title("Feed Forward Neural Network")
            elif op.MODEL_TYPE == "cnn":
                self.my_plot.set_title("Convolution Neural Network")

    def show_plot(self):
        if op.DO_SHOW:
            self.my_plot.legend(loc="lower right")
            plt.show()
