import argparse
import sys

current_frame = sys.argv[0].split('/')[-1]
current_frame = current_frame.split('.py')[0]

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments(is_training=False):
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is vectors_dataset_parsing_{num_of_fold})"
                                                    "\nUseAge : python training.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python training.py -closed 1\n\n")
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is svm)"
                                                  "\nUseAge : python training.py -model (svm|ffnn|cnn)\n\n")
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python training.py -epoch 20000\n\n")
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python training.py -hidden 2 (non-linear)\n\n")
    parser.add_argument("-learn", "--learn", help="set a learning rate for training (default is 0.001)"
                                                  "\nUseAge : python training.py -learn 0.01\n\n")
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python training.py -show 1 (True)\n\n")
    parser.add_argument("-dir", "--dir", help="set directory name by distinction (default is Null)"
                                              "\nUseAge : python training.py -dir 'dir_name'\n\n")
    if not is_training:
        parser.add_argument("-plot", "--plot", help="show plot (default is 0)"
                                                    "\nUseAge : python predict.py -plot 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


if current_frame == "training":
    args = get_arguments(is_training=True)
else:
    args = get_arguments()

# LOAD options #
READ_VECTOR = False
vector_path = "modeling/vectors/"
vector_name = "vectors_dataset_parsing_5"

if args.vector:
    READ_VECTOR = args.vector
else:
    READ_VECTOR = vector_path + vector_name

# Training options #
IS_CLOSED = False
USE_W2V = False
MODEL_TYPE = "svm"

# Parameter options #
EPOCH = 2000
NUM_HIDDEN_LAYER = 0
NUM_HIDDEN_DIMENSION = 0
LEARNING_RATE = 0.0001

# SHOW options #
DO_SHOW = False
DO_SHOW_PLOT = False

if args.closed:
    try:
        closed = int(args.closed)
    except ValueError:
        print("\nInput Error type of closed option!\n")
        exit(-1)
    else:
        if closed != 1 and closed != 0:
            print("\nInput Error closed option!\n")
            exit(-1)
        if closed == 1:
            IS_CLOSED = True
        else:
            IS_CLOSED = False

if args.model:
    MODEL_TYPE = args.model
    if MODEL_TYPE != "ffnn" and MODEL_TYPE != "svm" and MODEL_TYPE != "cnn":
        print("\nInput Error model option! (You must input (svm|ffnn|cnn))\n")
        exit(-1)

if args.epoch:
    try:
        EPOCH = int(args.epoch)
    except ValueError:
        print("\nInput Error type of epoch option!\n")
        exit(-1)
    else:
        if EPOCH < 100:
            print("\nInput Error epoch option!\n")
            exit(-1)

if args.hidden:
    try:
        NUM_HIDDEN_LAYER = int(args.hidden)
    except ValueError:
        print("\nInput Error type of hidden option!\n")
        exit(-1)
    else:
        if NUM_HIDDEN_LAYER < 0:
            print("\nInput Error hidden option!\n")
            exit(-1)

if args.learn:
    try:
        LEARNING_RATE = float(args.learn)
    except ValueError:
        print("\nInput Error type of learn option!\n")
        exit(-1)
    else:
        if LEARNING_RATE > 1 or LEARNING_RATE < 0.0000001:
            print("\nInput Error Boundary of learn option!\n")
            exit(-1)

if args.show:
    try:
        DO_SHOW = int(args.show)
    except ValueError:
        print("\nInput Error type of show option!\n")
        exit(-1)
    else:
        if DO_SHOW != 1 and DO_SHOW != 0:
            print("\nInput Error show option!\n")
            exit(-1)

if current_frame == "predict":
    if args.plot:
        try:
            DO_SHOW_PLOT = int(args.plot)
        except ValueError:
            print("\nInput Error type of show option!\n")
            exit(-1)
        else:
            if DO_SHOW_PLOT != 1 and DO_SHOW_PLOT != 0:
                print("\nInput Error show option!\n")
                exit(-1)

# SAVE options #
if args.dir:
    SAVE_DIR_NAME = args.dir + "/"
else:
    # SAVE_DIR_NAME = (VECTOR_NAME)_(MODEL)_h_(NUM_OF_HIDDEN)_e_(EPOCH)_lr_(LEARNING_RATE)
    SAVE_DIR_NAME = READ_VECTOR.split('/')[-1] + "_" + \
                    MODEL_TYPE + "_h_" + str(NUM_HIDDEN_LAYER) + "_e_" + str(EPOCH) + "_lr_" + str(LEARNING_RATE) + "/"


def show_options():
    if IS_CLOSED:
        print("\n\n========== CLOSED DATA SET ==========\n")
    else:
        print("\n\n========== OPENED DATA SET ==========\n")

    if USE_W2V:
        print("Using word2vec\n")
    else:
        print("Not using word2vec\n")

    if MODEL_TYPE != "svm":
        print("# of hidden layers -", NUM_HIDDEN_LAYER)
        print("Learning Rate -", LEARNING_RATE)
        print("# of EPOCH -", EPOCH)
