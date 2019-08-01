import argparse
from DMP.dataset.variables import get_num_of_features

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-base", "--base", help="set a name of csv file for base to encode(default is None)"
                                                "\nthe File must be in dataset/parsing dictionary"
                                                "\nUseAge : python encoding.py -base 'B'\n\n",
                        default="", type=str)
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'parsing'.csv"
                                                  "\nthe File must be in dataset/parsing dictionary"
                                                  "\nUseAge : python encoding.py -input 'Name'\n\n")
    parser.add_argument("-output", "--output", help="set vector file name to train or predict (default is 'model')"
                                                    "\nUseAge : python encoding.py -output 'NAME_OF_VECTOR'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\nUseAge : python encoding.py -target 'symptom'\n\n")
    parser.add_argument("-w2v", "--w2v", help="using word2vec or not (default is 0)"
                                              "\nUseAge : python encoding.py -w2v 1 (True)\n\n")
    parser.add_argument("-ver", "--version", help="set a version for vectorization (Default is 1)"
                                                  "\n1 - make vector for Training"
                                                  "\n2 - make vector for Feature Selection"
                                                  "\nUseAge : python encoding.py -ver 1\n\n")
    parser.add_argument("-fs", "--fs", help="set a json file to select important features(default is None)"
                                            "\nUseAge : python encoding.py -fs 'F'\n\n")
    parser.add_argument("-n_features", "--n_features",
                        help="set a number of importance features for making vectors(default is k)"
                             "\nUseAge : python encoding.py -n_features 'N'\n\n")
    parser.add_argument("-softmax", "--softmax", help="set a output nodes for cross entropy(default is 0)"
                                                      "\n0 - make just 1 output node (prediction)"
                                                      "\n1 - make output node for softmax cross entropy"
                                                      "\nUseAge : python encoding.py -cross_entropy 'C'\n\n",
                        default=0, type=int)
    _args = parser.parse_args()

    return _args


args = get_arguments()

EXTENSION_FILE = ".csv"
SAVE_FILE_TOTAL = "parsing"
SAVE_FILE_TRAIN = "parsing_train" + EXTENSION_FILE
SAVE_FILE_VALID = "parsing_valid" + EXTENSION_FILE
SAVE_FILE_TEST = "parsing_test" + EXTENSION_FILE
COLUMN_TARGET = str()
FILE_VECTOR = "model"
LOG_PATH = "modeling/fsResult/"

USE_W2V = False
VERSION = 1

NUM_OF_FEATURES = get_num_of_features()
NUM_OF_IMPORTANT = NUM_OF_FEATURES

DO_CROSS_ENTROPY = args.softmax

if DO_CROSS_ENTROPY != 1 and DO_CROSS_ENTROPY != 0:
    print("\nInput Error softmax option!\n")
    exit(-1)

if args.input:
    SAVE_FILE_TOTAL = args.input
    
if args.output:
    FILE_VECTOR = args.output

if args.target:
    COLUMN_TARGET = args.target

    if COLUMN_TARGET == "b":
        COLUMN_TARGET = "CR"
        symptom = "bacteremia"
    elif COLUMN_TARGET == "s":
        COLUMN_TARGET = "CU"
        symptom = "sepsis"
    elif COLUMN_TARGET == "p":
        COLUMN_TARGET = "CS"
        symptom = "pneumonia"
    else:
        symptom = str()

    if symptom:
        SAVE_FILE_TRAIN = SAVE_FILE_TOTAL + "_" + symptom + "_train" + EXTENSION_FILE
        SAVE_FILE_VALID = SAVE_FILE_TOTAL + "_" + symptom + "_valid" + EXTENSION_FILE
        SAVE_FILE_TEST = SAVE_FILE_TOTAL + "_" + symptom + "_test" + EXTENSION_FILE
        SAVE_FILE_TOTAL = SAVE_FILE_TOTAL + "_" + symptom + EXTENSION_FILE
else:
    SAVE_FILE_TRAIN = SAVE_FILE_TOTAL + "_train" + EXTENSION_FILE
    SAVE_FILE_VALID = SAVE_FILE_TOTAL + "_valid" + EXTENSION_FILE
    SAVE_FILE_TEST = SAVE_FILE_TOTAL + "_test" + EXTENSION_FILE
    SAVE_FILE_TOTAL += EXTENSION_FILE

if args.base:
    SAVE_FILE_TOTAL = args.base + EXTENSION_FILE

if args.w2v:
    try:
        USE_W2V = int(args.w2v)
    except ValueError:
        print("\nInput Error type of w2v option!\n")
        exit(-1)
    else:
        if USE_W2V != 1 and USE_W2V != 0:
            print("\nInput Error Boundary of w2v option!\n")
            exit(-1)
        if USE_W2V == 1:
            USE_W2V = True
        else:
            USE_W2V = False

if args.version:
    try:
        VERSION = int(args.version)
    except ValueError:
        print("\nInput Error type of version option!\n")
        exit(-1)
    else:
        if VERSION != 1 and VERSION != 2:
            print("\nInput Error Boundary of version option!\n")
            exit(-1)

if args.fs:
    LOG_NAME = args.fs
else:
    LOG_NAME = str()

if args.n_features:
    try:
        NUM_OF_IMPORTANT = int(args.n_features)
    except ValueError:
        print("\nInput Error type of n option!\n")
        exit(-1)
    else:
        if NUM_OF_IMPORTANT == -1:
            NUM_OF_IMPORTANT = NUM_OF_FEATURES
        elif NUM_OF_IMPORTANT < 1 or NUM_OF_IMPORTANT > NUM_OF_FEATURES:
            print("\nInput Error Boundary of n option!\n")
            exit(-1)
