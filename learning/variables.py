# svm kernel
SVM_KERNEL = "linear"

# dimension of hidden layer
NUM_HIDDEN_DIMENSION = 0
RATIO_HIDDEN = 2

# path of directory for loading and storing
PATH_LOGS = "logs/"
PATH_TENSOR = "modeling/save/"
PATH_RESULT = "analysis/"
FILE_OF_TRAINING_TIME = "run_time"

# name of tensor
NAME_HYPO = "hypothesis"
NAME_PREDICT = "predict"
NAME_X = "tf_x"
NAME_Y = "tf_y"
NAME_PROB = "keep_prob"
NAME_ACC = "accuracy"
NAME_LEARNING_RATE = "learning_rate"
NAME_HIDDEN = "num_of_hidden"
NAME_SCOPE_COST = "cost"
NAME_SCOPE_PREDICT = "prediction"
NAME_FC = "fc"
NAME_FC_W = "W"
NAME_FC_B = "b"

# key of target
KEY_TRAIN = "training"
KEY_VALID = "validation"
KEY_TEST = "test"
KEY_MORTALITY = "mortality"
KEY_IMMORTALITY = "alive"
KEY_TOTAL = "total"

# key of measure of process
KEY_PRECISION = "precision"
KEY_RECALL = "recall"
KEY_F1 = "f1"
KEY_ACCURACY = "accuracy"
KEY_AUC = "auc"

NUM_OF_K_FOLD = 5

# gray scale for convolution neural network
GRAY_SCALE = 255

# initialize a image size for convolution neural networks
# INITIAL_IMAGE_SIZE = 36
INITIAL_IMAGE_SIZE = 50

# drop out ratio
KEEP_PROB = 0.7

# if valid loss increase X in a row
NUM_OF_LOSS_OVER_FIT = 5

# save every X epoch
NUM_OF_SAVE_EPOCH = 1

# mini-batch size
BATCH_SIZE = 16

# Image options #
# set CT image size
IMAGE_RESIZE = 224
DO_NORMALIZE = False
USE_EARLY_STOPPING = True
USE_EARLY_STOPPING_WITH_VAL_ACC = True
