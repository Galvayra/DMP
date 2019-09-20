from .variables import *
from .score import MyScore
from os import path
import os
import shutil
import sys
import json
from DMP.modeling.variables import MODELING_PATH, TF_RECORD_PATH
from DMP.modeling.tfRecorder import TfRecorder, EXTENSION_OF_TF_RECORD
import tensorflow as tf
import numpy as np

current_script = sys.argv[0].split('/')[-1]
if current_script == "training.py":
    from DMP.utils.arg_training import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE, \
        READ_VECTOR
elif current_script == "predict.py":
    from DMP.utils.arg_predict import DO_SHOW, DO_DELETE, TENSOR_DIR_NAME, EPOCH, NUM_HIDDEN_LAYER, LEARNING_RATE, \
        READ_VECTOR
elif current_script == "fine_tuning.py":
    from DMP.utils.arg_fine_tuning import DO_SHOW, NUM_HIDDEN_LAYER, EPOCH, DO_DELETE, TENSOR_DIR_NAME, LEARNING_RATE, \
        READ_VECTOR
elif current_script == "predict_tfRecord.py":
    from DMP.utils.arg_predict_tfRecord import DO_SHOW, DO_DELETE, TENSOR_DIR_NAME, EPOCH, NUM_HIDDEN_LAYER, \
        LEARNING_RATE, READ_VECTOR


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
        self.name_of_log = str()
        self.name_of_tensor = str()
        self.is_cross_valid = is_cross_valid
        self.init_log_and_tensor()
        self.do_show = DO_SHOW
        project_path = path.dirname(path.dirname(path.abspath(__file__))) + "/"
        self.tf_record_path = project_path + MODELING_PATH + TF_RECORD_PATH + READ_VECTOR.split('/')[-1] + "/"
        self.tf_recorder = TfRecorder(self.tf_record_path)

        if current_script == "training.py" or current_script == "fine_tuning.py":
            if DO_SHOW:
                print("\n=============== hyper-parameters ===============")
                if self.is_cross_valid:
                    print("Epoch -", self.best_epoch)
                else:
                    print("Epoch - unlimited")
                print("Learning Rate -", self.learning_rate)
                print("Mini-batch Size -", BATCH_SIZE, '\n\n')

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
        if self.is_cross_valid:
            name_of_log = self.name_of_log + "fold_" + str(self.num_of_fold)
            os.mkdir(name_of_log)
        else:
            name_of_log = self.name_of_log

        if DO_SHOW:
            print("\n======== Directory for Saving ========")
            print("   Log File -", name_of_log)

    def set_name_of_tensor(self):
        if self.is_cross_valid:
            name_of_tensor = self.name_of_tensor + "fold_" + str(self.num_of_fold)
            os.mkdir(name_of_tensor)
        else:
            name_of_tensor = self.name_of_tensor

        if DO_SHOW:
            print("Tensor File -", name_of_tensor, "\n\n\n")

    def get_name_of_tensor(self):
        if self.is_cross_valid:
            return self.name_of_tensor + "fold_" + str(self.num_of_fold)
        else:
            return self.name_of_tensor

    def init_tf_record_tensor(self, key, is_test=False):
        tf_record_path = self.tf_record_path + key + EXTENSION_OF_TF_RECORD
        print("Initialize tfRecord -", self.tf_record_path + key + EXTENSION_OF_TF_RECORD)

        tf_recode = self.tf_recorder.get_img_from_tf_records(tf_record_path)
        if is_test:
            tf_recode = tf_recode.repeat(1)
        else:
            if self.is_cross_valid:
                tf_recode = tf_recode.repeat(EPOCH)
            else:
                tf_recode = tf_recode.repeat()

        tf_recode = tf_recode.batch(BATCH_SIZE)

        return tf_recode

    def get_total_batch(self, key, is_get_image=False):
        x_data, y_data = list(), list()

        tf_train_summary = self.init_tf_record_tensor(key=key, is_test=True)
        iterator_summary = tf_train_summary.make_initializable_iterator()
        next_summary_element = iterator_summary.get_next()

        with tf.Session() as sess:
            sess.run(iterator_summary.initializer)
            try:
                while True:
                    x_batch, y_batch, x_img, tensor_name = sess.run(next_summary_element)

                    for x, x_img, y in zip(x_batch, x_img, y_batch):
                        if is_get_image:
                            x_data.append(x_img)
                        else:
                            x_data.append(x)
                        y_data.append(y)

            except tf.errors.OutOfRangeError:
                x_data = np.array(x_data)
                y_data = np.array(y_data)

        return x_data, y_data

    def save_process_time(self):
        with open(self.name_of_log + FILE_OF_TRAINING_TIME, 'w') as outfile:
            json.dump(self.show_process_time(), outfile, indent=4)

    def clear_tensor(self):
        self.tf_x = None
        self.tf_y = None


class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        self.is_stop = False

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
