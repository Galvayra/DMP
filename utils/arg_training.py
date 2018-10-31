import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is vectors_dataset_parsing_{num_of_fold})"
                                                    "\nUseAge : python training.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is svm)"
                                                  "\nUseAge : python training.py -model (ffnn|cnn)\n\n")
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python training.py -epoch 20000\n\n")
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python training.py -hidden 2 (non-linear)\n\n")
    parser.add_argument("-learn", "--learn", help="set a learning rate for training (default is 0.001)"
                                                  "\nUseAge : python training.py -learn 0.01\n\n")
    parser.add_argument("-log", "--log", help="set directory name for log and tensor (default is Null)"
                                              "\nUseAge : python training.py -dir 'dir_name'\n\n")
    parser.add_argument("-delete", "--delete", help="set whether SAVE_DIR will be delete (default is 1)"
                                                    "\nIf you already have dir for saving, delete it and then save"
                                                    "\nIf you set False, It will be stopped before training"
                                                    "\nUseAge : python training.py -delete 0 (False)\n\n")
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python training.py -show 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


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
USE_W2V = False
MODEL_TYPE = "ffnn"

# Parameter options #
EPOCH = 2000
NUM_HIDDEN_LAYER = 0
NUM_HIDDEN_DIMENSION = 0
LEARNING_RATE = 0.0001

# SHOW options #
DO_SHOW = False

# SAVE options #
LOG_DIR_NAME = str()
DO_DELETE = False


if args.model:
    MODEL_TYPE = args.model
    if MODEL_TYPE != "ffnn" and MODEL_TYPE != "cnn":
        print("\nInput Error model option! (You must input (ffnn|cnn))\n")
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

# SAVE options #
if args.log:
    LOG_DIR_NAME = args.log + "/"
else:
    LOG_DIR_NAME = READ_VECTOR.split('/')[-1] + "_" + \
                    MODEL_TYPE + "_h_" + str(NUM_HIDDEN_LAYER) + "_e_" + str(EPOCH) + "_lr_" + str(LEARNING_RATE) + "/"

if args.delete:
    try:
        DO_DELETE = int(args.delete)
    except ValueError:
        print("\nInput Error type of delete option!\n")
        exit(-1)
    else:
        if DO_DELETE != 1 and DO_DELETE != 0:
            print("\nInput Error delete option!\n")
            exit(-1)


def show_options():
    if DO_SHOW:
        if USE_W2V:
            print("Using word2vec\n")
        else:
            print("Not using word2vec\n")

        print("model -", MODEL_TYPE)
        print("# of hidden layers -", NUM_HIDDEN_LAYER)
        print("Learning Rate -", LEARNING_RATE)
        print("# of EPOCH -", EPOCH, "\n\n")
