import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python encoding.py -closed 1\n\n")
    parser.add_argument("-id", "--identify", help="set id for separating training sets (default is None)"
                                                  "\nUseAge : python encoding.py -id string\n\n")
    parser.add_argument("-svm", "--svm", help="training use support vector machine (default is 0)"
                                              "\nUseAge : python training.py -svm 1 -w2v 1\n\n")
    parser.add_argument("-w2v", "--word2v", help="using word2vec (default is 1)"
                                                 "\nUseAge : python training.py -w2v 1 (True)"
                                                 "\n         python training.py -w2v 0 (False)\n\n")
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python training.py -epoch \n\n")
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python training.py -hidden 2 (non-linear)\n\n")
    parser.add_argument("-show", "--show", help="show plot (default is 0)"
                                                "\nUseAge : python training.py -show 1 (True)\n\n")
    parser.add_argument("-train", "--train", help="set vector file name to train or predict (default is Null)"
                                                  "\nUseAge : python training.py -train 'file_name'\n\n")
    parser.add_argument("-dir", "--dir", help="set directory name by distinction (default is Null)"
                                              "\nUseAge : python training.py -dir 'dir_name'\n\n")
    _args = parser.parse_args()

    return _args


NUM_FOLDS = 5
# NUM_FOLDS = 1
RATIO = 10

IS_CLOSED = False

args = get_arguments()

if not args.closed:
    IS_CLOSED = False
else:
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

if not args.svm:
    DO_SVM = False
else:
    try:
        DO_SVM = int(args.svm)
    except ValueError:
        print("\nInput Error type of test option!\n")
        exit(-1)
    else:
        if DO_SVM != 1 and DO_SVM != 0:
            print("\nInput Error test option!\n")
            exit(-1)

if not args.word2v:
    USE_W2V = True
else:
    try:
        USE_W2V = int(args.word2v)
    except ValueError:
        print("\nInput Error type of word2v option!\n")
        exit(-1)
    else:
        if USE_W2V != 1 and USE_W2V != 0:
            print("\nInput Error word2v option!\n")
            exit(-1)

if not args.show:
    DO_SHOW = 0
else:
    try:
        DO_SHOW = int(args.show)
    except ValueError:
        print("\nInput Error type of show option!\n")
        exit(-1)
    else:
        if DO_SHOW != 1 and DO_SHOW != 0:
            print("\nInput Error show option!\n")
            exit(-1)

if not args.identify:
    USE_ID = str()
else:
    USE_ID = args.identify + "#"

if not args.epoch:
    EPOCH = 2000
else:
    try:
        EPOCH = int(args.epoch)
    except ValueError:
        print("\nInput Error type of epoch option!\n")
        exit(-1)
    else:
        if EPOCH < 100:
            print("\nInput Error epoch option!\n")
            exit(-1)

if not args.hidden:
    NUM_HIDDEN_LAYER = 0
else:
    try:
        NUM_HIDDEN_LAYER = int(args.hidden)
    except ValueError:
        print("\nInput Error type of hidden option!\n")
        exit(-1)
    else:
        if NUM_HIDDEN_LAYER < 0:
            print("\nInput Error hidden option!\n")
            exit(-1)

NUM_HIDDEN_DIMENSION = 0

if not args.train:
    FILE_VECTOR = str()
else:
    FILE_VECTOR = args.train

if not args.dir:
    SAVE_DIR_NAME = str()
else:
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
