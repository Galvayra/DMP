from DMP.dataset.variables import columns_dict
from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is vectors_dataset_parsing_{num_of_fold})"
                                                    "\nUseAge : python predict.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-image_dir", "--image_dir", help="set a path of image directory (default is None)"
                                                          "It is only apply to use cnn model"
                                                          "\nUseAge : python predict.py -image_dir 'path'\n\n")
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is svm)"
                                                  "\nUseAge : python predict.py -model (svm|ffnn|cnn)\n\n")
    parser.add_argument("-feature", "--feature", help="set a feature to predict (default is merge(all))"
                                                      "\nUseAge : python predict.py -feature 'TYPE_OF_FEATURE'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom"
                                                    "\n(s, sepsis), (p, pneumonia), (b, bacteremia)"
                                                    "\nUseAge : python predict.py -target s(sepsis)\n\n")
    parser.add_argument("-tensor_dir", "--tensor_dir", help="set directory name for log and tensor (default is Null)"
                                                            "\nUseAge : python predict.py -tensor_dir 'dir_name'\n\n")
    parser.add_argument("-save", "--save", help="save a score to csv file (default is 'LOG_NAME')"
                                                "\nUseAge : python predict.py -save 'NAME' (in analysis dir)\n\n")
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python predict.py -show 1 (True)\n\n")
    parser.add_argument("-plot", "--plot", help="set a option for visualization (default is 0)"
                                                "\nUseAge : python predict.py -plot 1 (True)\n\n")
    parser.add_argument("-ver", "--version", help="set a version for training (Default is 1)"
                                                  "\n1 - 5-cross validation"
                                                  "\n2 - hyper-param optimization using valid set"
                                                  "\nUseAge : python predict.py -ver 1\n\n")

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
TYPE_OF_MODEL = "svm"
TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR
type_of_features = [TYPE_OF_FEATURE] + [type_of_column for type_of_column in columns_dict]
IMAGE_PATH = str()

# Target options #
COLUMN_TARGET = False
COLUMN_TARGET_NAME = str()

# SHOW options #
DO_SHOW = False
DO_SHOW_PLOT = False

# SAVE options #
TENSOR_DIR_NAME = str()
SAVE_DIR_NAME = str()
DO_DELETE = False

VERSION = 1

EPOCH = int()
NUM_HIDDEN_LAYER = int()
LEARNING_RATE = float()

if args.image_dir:
    IMAGE_PATH = args.image_dir
    if not os.path.isdir(IMAGE_PATH):
        print("\nFileNotFoundError image_dir option!\n")
        exit(-1)
    else:
        print("Success to read image files -", IMAGE_PATH, '\n\n')

if args.model:
    TYPE_OF_MODEL = args.model
    if TYPE_OF_MODEL != "ffnn" and TYPE_OF_MODEL != "cnn" and TYPE_OF_MODEL != "svm":
        print("\nInput Error model option! (You must input (svm|ffnn|cnn))\n")
        exit(-1)

if args.target:
    COLUMN_TARGET = args.target

    if COLUMN_TARGET == "b":
        COLUMN_TARGET_NAME = "bacteremia"
        COLUMN_TARGET = "CR"
    elif COLUMN_TARGET == "s":
        COLUMN_TARGET_NAME = "sepsis"
        COLUMN_TARGET = "CU"
    elif COLUMN_TARGET == "p":
        COLUMN_TARGET_NAME = "pneumonia"
        COLUMN_TARGET = "CS"
    else:
        print("\nInput Error target option!")
        print("Please input target (s|p|b)")
        exit(-1)

if args.feature:
    TYPE_OF_FEATURE = args.feature

    if TYPE_OF_FEATURE not in type_of_features:
        print("\nInput Error feature option!")
        print("You must input -", type_of_features)
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
if args.tensor_dir:
    TENSOR_DIR_NAME = args.tensor_dir

if TYPE_OF_FEATURE != KEY_NAME_OF_MERGE_VECTOR:
    TENSOR_DIR_NAME += "_" + TYPE_OF_FEATURE + "/"
else:
    TENSOR_DIR_NAME += "/"

if args.save:
    SAVE_DIR_NAME = args.save
else:
    SAVE_DIR_NAME = TENSOR_DIR_NAME[:-1]

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


def show_options():
    if DO_SHOW:
        if IS_CLOSED:
            print("\n\n========== CLOSED DATA SET ==========\n")
        else:
            print("\n\n========== OPENED DATA SET ==========\n")

        if USE_W2V:
            print("Using word2vec\n")
        else:
            print("Not using word2vec\n")

        if COLUMN_TARGET_NAME:
            print("Target is -", COLUMN_TARGET_NAME, "\n")
        else:
            print("Target is All\n")

        if TYPE_OF_MODEL == "svm":
            print("\n\n--- Support Vector Machine    !! ---\n\n\n")
        else:
            if VERSION == 1:
                print("\n\n--- 5 Cross Validation !! ---\n\n\n")
            elif VERSION == 2:
                print("\n\n--- Optimize hyper-param using validation set !! ---\n\n\n")

        print("model -", TYPE_OF_MODEL, "\n\n")
