import matplotlib.pyplot as plt
import sys

current_script = sys.argv[0].split('/')[-1]

if current_script == "training.py":
    from DMP.utils.arg_training import TYPE_OF_MODEL
elif current_script == "predict.py" or current_script == "show_multi_plot.py":
    from DMP.utils.arg_predict import TYPE_OF_MODEL, DO_SHOW_PLOT

TOP_N = 10


class MyPlot:
    def __init__(self):
        self.__my_plot = None
        self.tpr, self.fpr = self.__init_plot()
        self.auc = float()
        self.tra_loss_list = list()
        self.val_loss_list = list()
        self.tra_acc_list = list()
        self.val_acc_list = list()

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

    def save_loss_plot(self, log_path, step_list):
        plt.plot(step_list, self.tra_loss_list, '-c', label='loss')
        plt.plot(step_list, self.val_loss_list, '-r', label='loss')
        plt.plot(step_list, self.tra_acc_list, '-y', label='accuracy')
        plt.plot(step_list, self.val_acc_list, '-m', label='accuracy')

        plt.xlabel("epoch")
        plt.legend(loc='upper left')
        plt.title("cost/accuracy graph")
        plt.savefig(log_path + "graph.png")
        print("\nSuccess to save graph -", log_path + "graph.png")

    def set_plot(self, k=None, fpr=None, tpr=None, title=None):
        if DO_SHOW_PLOT:
            if fpr:
                self.fpr = fpr
            if tpr:
                self.tpr = tpr

            if not title:
                self.my_plot.plot(self.fpr, self.tpr, alpha=0.3, label='ROC %d (AUC = %0.1f)' % (k, self.auc))
            else:
                self.my_plot.plot(self.fpr, self.tpr, alpha=0.5, label='%s' % title)

            self.tpr, self.fpr = self.__init_plot()

    def show_plot(self):
        if DO_SHOW_PLOT:
            self.my_plot.legend(loc="lower right")
            plt.show()

    @staticmethod
    def show_importance(feature_importance):
        top_features = feature_importance[:TOP_N]

        x = ['\n'.join(f[1][1].split()) for f in top_features]
        y = [f[2] for f in top_features]

        ax = plt.subplot(111, xlabel='features', ylabel='importance', title='Top 10 high importance features')

        # [ax.title, ax.xaxis.label, ax.yaxis.label] +
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(7)

        plt.bar(x, y)
        plt.show()
