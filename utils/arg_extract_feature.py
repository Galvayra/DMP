from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
from DMP.dataset.variables import get_num_of_features
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set loading vector file name to get importance of features"
                                                    "\nUseAge : python extract_feature.py -vector 'V'\n\n")
    parser.add_argument("-output", "--output", help="set saving a result of feature selection"
                                                    "\n(default is 'fsResult')"
                                                    "\nUseAge : python extract_feature.py -output 'O'\n\n")
    parser.add_argument("-ntree", "--ntree", help="set a number of tree in random forest (default is 400)"
                                                  "\nUseAge : python extract_feature.py -ntree 400\n\n")
    parser.add_argument("-show", "--show", help="show importance features (default is 0)"
                                                "\nUseAge : python extract_feature.py -show 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# LOAD options #
DUMP_PATH = "modeling/vectors/"
DUMP_FILE = "model"
SAVE_PATH = "modeling/fsResult/"
READ_VECTOR = DUMP_PATH + DUMP_FILE
SAVE_LOG_NAME = SAVE_PATH + "fsResult"

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR

DO_SHOW = False
NUM_OF_FEATURES = get_num_of_features()

# Random Forest Option #
NUM_OF_TREE = 400


if args.vector:
    READ_VECTOR = args.vector
else:
    print("\nPlease use vector option!\n")
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

if args.ntree:
    try:
        NUM_OF_TREE = int(args.ntree)
    except ValueError:
        print("\nInput Error type of ntree option!\n")
        exit(-1)
    else:
        if NUM_OF_TREE < 10 or NUM_OF_TREE > 10000:
            print("\nInput Error ntree option!\n")
            exit(-1)

if args.output:
    SAVE_LOG_NAME = SAVE_PATH + args.output


def show_options():
    if DO_SHOW:
        print("# of tree in random forest -", NUM_OF_TREE)
