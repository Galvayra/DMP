from .variables import *
from .score import MyScore
from os import path
import os
import shutil
import sys
from DMP.modeling.variables import MODELING_PATH, TF_RECORD_PATH
from DMP.modeling.tfRecorder import TfRecorder, EXTENSION_OF_TF_RECORD
import tensorflow as tf

if sys.argv[0].split('/')[-1] == "training.py":
    from DMP.utils.arg_training import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE, \
        READ_VECTOR
elif sys.argv[0].split('/')[-1] == "predict.py":
    from DMP.utils.arg_predict import DO_SHOW, DO_DELETE, TENSOR_DIR_NAME, EPOCH, NUM_HIDDEN_LAYER, LEARNING_RATE, \
        READ_VECTOR
elif sys.argv[0].split('/')[-1] == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE, \
        READ_VECTOR


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
        project_path = path.dirname(path.dirname(path.abspath(__file__))) + "/"
        self.tf_record_path = project_path + MODELING_PATH + TF_RECORD_PATH + READ_VECTOR.split('/')[-1] + "/"
        self.tf_recorder = TfRecorder(self.tf_record_path)

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

    def init_tf_record_tensor(self, key, is_test=False):
        tf_record_path = self.tf_record_path + key + str(self.num_of_fold) + EXTENSION_OF_TF_RECORD

        tf_recode = self.tf_recorder.get_img_from_tf_records(tf_record_path)
        if not is_test:
            tf_recode = tf_recode.repeat(10)
        tf_recode = tf_recode.batch(BATCH_SIZE)

        return tf_recode

    def clear_tensor(self):
        self.tf_x = None
        self.tf_y = None
