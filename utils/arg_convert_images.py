from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set loading vector file name to train or predict"
                                                    "\n(default is 'model')"
                                                    "\nUseAge : python extract_feature.py -vector 'V'\n\n")
    parser.add_argument("-output", "--output", help="set saving vector file name to train or predict"
                                                    "\n(default is 'vector'+'_new')"
                                                    "\nUseAge : python extract_feature.py -output 'O'\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# LOAD options #
DUMP_PATH = "modeling/vectors/"
DUMP_FILE = "model"
DUMP_IMAGE = "images/"
DUMP_TRAIN = "train/"
DUMP_VALID = "train/"
DUMP_TEST = "train/"
READ_VECTOR = DUMP_PATH + DUMP_FILE
SAVE_VECTOR = READ_VECTOR + "_new"

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR


if args.vector:
    READ_VECTOR = args.vector

if args.output:
    SAVE_VECTOR = DUMP_PATH + args.output


def show_options():
    pass
