import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is vectors_dataset_parsing_{num_of_fold})"
                                                    "\nUseAge : python predict.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python predict.py -closed 1\n\n")
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is svm)"
                                                  "\nUseAge : python predict.py -model (svm|ffnn|cnn)\n\n")
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python predict.py -epoch 20000\n\n")
    parser.add_argument("-log", "--log", help="set directory name for log and tensor (default is Null)"
                                              "\nUseAge : python predict.py -dir 'dir_name'\n\n")
    parser.add_argument("-save", "--save", help="save a score to csv file (default is 'LOG_NAME')"
                                                "\nUseAge : python predict.py -save 'NAME' (in analysis dir)\n\n")
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python predict.py -show 1 (True)\n\n")
    parser.add_argument("-plot", "--plot", help="set a option for visualization (default is 0)"
                                                "\nUseAge : python predict.py -plot 1 (True)\n\n")

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
IS_CLOSED = False
USE_W2V = False
MODEL_TYPE = "svm"

# Parameter options #
EPOCH = 2000

# SHOW options #
DO_SHOW = False
DO_SHOW_PLOT = False

# SAVE options #
LOG_DIR_NAME = str()
SAVE_DIR_NAME = str()
DO_DELETE = False

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

if args.save:
    SAVE_DIR_NAME = args.save
else:
    SAVE_DIR_NAME = LOG_DIR_NAME[:-1]

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


def show_options():
    if IS_CLOSED:
        print("\n\n========== CLOSED DATA SET ==========\n")
    else:
        print("\n\n========== OPENED DATA SET ==========\n")

    if USE_W2V:
        print("Using word2vec\n")
    else:
        print("Not using word2vec\n")

    print("model -", MODEL_TYPE)
    if MODEL_TYPE != "svm":
        print("# of EPOCH -", EPOCH)

    print("\n\n")
