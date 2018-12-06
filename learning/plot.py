import matplotlib.pyplot as plt
import sys

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL
elif sys.argv[0].split('/')[-1] == "predict.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL, DO_SHOW_PLOT


class MyPlot:
    def __init__(self):
        self.__my_plot = None
        self.tpr, self.fpr = self.__init_plot()

    @property
    def my_plot(self):
        return self.__my_plot
    
    @staticmethod
    def __init_plot():
        return None, None

    def init_plot(self):
        if DO_SHOW_PLOT:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            self.__my_plot = plt.subplot2grid((2, 2), (0, 0))
            self.my_plot.set_ylabel("Sensitivity")
            self.my_plot.set_xlabel("100 - Specificity")

            if TYPE_OF_MODEL == "svm":
                self.my_plot.set_title("Support Vector Machine")
            elif TYPE_OF_MODEL == "ffnn":
                self.my_plot.set_title("Feed Forward Neural Network")
            elif TYPE_OF_MODEL == "cnn":
                self.my_plot.set_title("Convolution Neural Network")

    def set_plot(self, target):
        if DO_SHOW_PLOT:
            # self.my_plot.plot(self.fpr, self.tpr, alpha=0.3,
            #                   label='%s AUC = %0.1f' % (target, self.score_dict[target][KEY_AUC]))
            self.my_plot.plot(self.fpr, self.tpr, alpha=0.3, label='AUC of %s' % target)
            self.tpr, self.fpr = self.__init_plot()

    def show_plot(self):
        if DO_SHOW_PLOT:
            self.my_plot.legend(loc="lower right")
            plt.show()
