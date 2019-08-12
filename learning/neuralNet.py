from .variables import *
from .score import MyScore
import os
import shutil
import sys

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE
elif sys.argv[0].split('/')[-1] == "predict.py":
    from DMP.utils.arg_predict import DO_SHOW, DO_DELETE, TENSOR_DIR_NAME, EPOCH, NUM_HIDDEN_LAYER, LEARNING_RATE
elif sys.argv[0].split('/')[-1] == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE


class TensorModel(MyScore):
    def __init__(self, is_cross_valid=True):
        super().__init__()
        self.tf_x = None
        self.tf_y = None
        self.keep_prob = None
        self.hypothesis = None
        self.best_epoch = EPOCH
        self.num_of_dimension = int()
        self.num_of_hidden = NUM_HIDDEN_LAYER
        # self.num_of_hidden = int()
        self.learning_rate = LEARNING_RATE
        # self.learning_rate = float()
        self.loss_list = list()
        self.name_of_log = str()
        self.name_of_tensor = str()
        self.is_cross_valid = is_cross_valid
        self.init_log_and_tensor()
        self.do_show = DO_SHOW

    def init_log_and_tensor(self):
        self.name_of_log = PATH_LOGS + TENSOR_DIR_NAME
        self.name_of_tensor = PATH_TENSOR + TENSOR_DIR_NAME

        if DO_DELETE:
            if os.path.isdir(self.name_of_log):
                shutil.rmtree(self.name_of_log)
            os.mkdir(self.name_of_log)

            if os.path.isdir(self.name_of_tensor):
                shutil.rmtree(self.name_of_tensor)
            os.mkdir(self.name_of_tensor)

    def set_name_of_log(self):
        name_of_log = self.name_of_log + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_log)

        if DO_SHOW:
            print("======== Directory for Saving ========")
            print("   Log File -", name_of_log)

    def set_name_of_tensor(self):
        name_of_tensor = self.name_of_tensor + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_tensor)

        if DO_SHOW:
            print("Tensor File -", name_of_tensor, "\n\n\n")

    def get_name_of_tensor(self):
        return self.name_of_tensor + "fold_" + str(self.num_of_fold)

    def clear_tensor(self):
        self.tf_x = None
        self.tf_y = None
