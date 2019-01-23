import matplotlib.pyplot as plt
import sys

DO_SHOW_PLOT = False


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
            self.my_plot.set_title("Inception v3")

    def set_plot(self, target):
        if DO_SHOW_PLOT:
            self.my_plot.plot(self.fpr, self.tpr, alpha=0.3, label='AUC of %s' % target)
            self.tpr, self.fpr = self.__init_plot()

    def show_plot(self):
        if DO_SHOW_PLOT:
            self.my_plot.legend(loc="lower right")
            plt.show()
