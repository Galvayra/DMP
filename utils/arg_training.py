import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is vectors_dataset_parsing_{num_of_fold})"
                                                    "\nUseAge : python training.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python training.py -closed 1\n\n")
    parser.add_argument("-fold", "--fold", help="set a number of k-fold (default is 5)"
                                                "\nUseAge : python training.py -fold 5\n\n")
    parser.add_argument("-id", "--identify", help="set id for separating training sets (default is None)"
                                                  "\nUseAge : python training.py -id string\n\n")
    parser.add_argument("-svm", "--svm", help="training use support vector machine (default is 0)"
                                              "\nUseAge : python training.py -svm 1 -w2v 1\n\n")
    # parser.add_argument("-w2v", "--word2v", help="using word2vec (default is 0)"
    #                                              "\nUseAge : python training.py -w2v 1 (True)"
    #                                              "\n         python training.py -w2v 0 (False)\n\n")
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python training.py -epoch \n\n")
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python training.py -hidden 2 (non-linear)\n\n")
    parser.add_argument("-show", "--show", help="show plot (default is 0)"
                                                "\nUseAge : python training.py -show 1 (True)\n\n")
    parser.add_argument("-dir", "--dir", help="set directory name by distinction (default is Null)"
                                              "\nUseAge : python training.py -dir 'dir_name'\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# Training options #
RATIO = 5
NUM_FOLDS = 5
IS_CLOSED = False
USE_W2V = False
DO_SVM = False

# Parameter options #
EPOCH = 2000
NUM_HIDDEN_LAYER = 0
NUM_HIDDEN_DIMENSION = 0

# SAVE options #
USE_ID = str()
SAVE_DIR_NAME = str()

if args.fold:
    try:
        NUM_FOLDS = int(args.fold)
    except ValueError:
        print("\nInput Error type of fold option!\n")
        exit(-1)
    else:
        if NUM_FOLDS < 1 or NUM_FOLDS > 10:
            print("\nInput Error a boundary of fold option!\n")
            exit(-1)

# LOAD options #
READ_VECTOR = False
vector_path = "modeling/vectors/"
vector_name = "vectors_dataset_parsing_" + str(NUM_FOLDS)

# SHOW options #
DO_SHOW = False

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

if args.identify:
    USE_ID = args.identify + "#"

if args.vector:
    READ_VECTOR = args.vector
else:
    READ_VECTOR = vector_path + vector_name

if args.svm:
    try:
        DO_SVM = int(args.svm)
    except ValueError:
        print("\nInput Error type of test option!\n")
        exit(-1)
    else:
        if DO_SVM != 1 and DO_SVM != 0:
            print("\nInput Error test option!\n")
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

if args.identify:
    USE_ID = args.identify + "#"

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

if args.dir:
    SAVE_DIR_NAME = args.dir + "/"


def show_options():
    if IS_CLOSED:
        print("\n\n========== CLOSED DATA SET ==========\n")
        print("k fold -", NUM_FOLDS)
    else:
        print("\n\n========== OPENED DATA SET ==========\n")
        print("k fold -", NUM_FOLDS)
        if NUM_FOLDS == 1:
            print("test ratio -", str(RATIO) + "%")

    if USE_W2V:
        print("\nUsing word2vec")
    else:
        print("\nNot using word2vec")

    print("num of hidden layers -", NUM_HIDDEN_LAYER)
    print("EPOCH -", EPOCH)
