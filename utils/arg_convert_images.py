from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set loading vector file name to train or predict"
                                                    "\n(default is 'model')"
                                                    "\nUseAge : python convert_images.py -vector 'V'\n\n")
    parser.add_argument("-output", "--output", help="set saving vector file name to train or predict"
                                                    "\n(default is 'vector'+'_new')"
                                                    "\nUseAge : python convert_images.py -output 'O'\n\n")
    parser.add_argument("-log", "--log", help="set name of log file (default is 'output option')"
                                              "\nUseAge : python convert_images.py -log 'L'\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# PATH options #
DUMP_PATH = "modeling/vectors/"
DUMP_FILE = "model"
DUMP_IMAGE = "dataset/images/dataset/"
DUMP_TRAIN = "train/"
DUMP_VALID = "valid/"
DUMP_TEST = "test/"
ALIVE_DIR = "alive/"
DEATH_DIR = "death/"
LOG_PATH = "dataset/images/log/"

READ_VECTOR = DUMP_PATH + DUMP_FILE
SAVE_VECTOR = DUMP_FILE + "_new"

DO_SHOW = True

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR


if args.vector:
    READ_VECTOR = args.vector

if args.output:
    SAVE_VECTOR = args.output

if args.log:
    LOG_NAME = args.log
else:
    LOG_NAME = SAVE_VECTOR


def show_options():
    pass
