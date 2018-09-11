import matplotlib.pyplot as plt
import DeepPredict.arguments as op


class MyPlot:
    def __init__(self):
        self.my_plot = None

    def init_plot(self):
        if op.DO_SHOW:

            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            self.my_plot = plt.subplot2grid((2, 2), (0, 0))
            self.my_plot.set_ylabel("Sensitivity")
            self.my_plot.set_xlabel("100 - Specificity")

            if op.DO_SVM:
                self.my_plot.set_title("Support Vector Machine")
            else:
                self.my_plot.set_title("Feed Forward Neural Network")

    def show_plot(self):
        if op.DO_SHOW:
            self.my_plot.legend(loc="lower right")
            plt.show()
