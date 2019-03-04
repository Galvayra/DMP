from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-version", "--version", help="set a convert version to save image files"
                                                      "\nversion 1 or 2 (default is '2')"
                                                      "\nUseAge : python convert_images.py -version 1\n\n",
                        default=2, type=int)
    parser.add_argument("-vector", "--vector", help="set loading vector file name to train or predict"
                                                    "\n(default is 'model')"
                                                    "\nUseAge : python convert_images.py -vector 'V'\n\n")
    parser.add_argument("-output", "--output", help="set saving vector file name to train or predict"
                                                    "\n(default is 'vector option')"
                                                    "\nUseAge : python convert_images.py -output 'O'\n\n")
    parser.add_argument("-log", "--log", help="set name of log file (default is 'output option')"
                                              "\nUseAge : python convert_images.py -log 'L'\n\n")
    parser.add_argument("-resize", "--resize", help="set a image size X (X by X pixel)"
                                                    "\n(default is 'None')"
                                                    "\nUseAge : python convert_images.py -resize 'X'\n\n")
    parser.add_argument("-transform", "--transform", help="build a image using transform like flip, rotate"
                                                          "\n(default is '0')"
                                                          "\nUseAge : python convert_images.py -transform 1\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# PATH options #
DUMP_PATH = "modeling/vectors/"
DUMP_FILE = "model"
DUMP_IMAGE = "dataset/images/image_data/"
DUMP_TRAIN = "train/"
DUMP_VALID = "valid/"
DUMP_TEST = "test/"
ALIVE_DIR = "alive/"
DEATH_DIR = "death/"
LOG_PATH = "dataset/images/log/"

READ_VECTOR = DUMP_PATH + DUMP_FILE
SAVE_VECTOR = DUMP_FILE + "_new"

DO_SHOW = True
DO_TRANSFORM = False

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR
CONVERT_VERSION = args.version


if args.vector:
    READ_VECTOR = args.vector

if args.output:
    SAVE_VECTOR = args.output
else:
    SAVE_VECTOR = READ_VECTOR.split('/')[-1]

if args.log:
    LOG_NAME = args.log
else:
    LOG_NAME = SAVE_VECTOR

if args.resize:
    try:
        IMAGE_SIZE = int(args.resize)
    except ValueError:
        print("\nValue Error of resize option!\n")
        exit(-1)
    else:
        if IMAGE_SIZE < 10 & IMAGE_SIZE > 4096:
            print("\nBoundary Error of resize option!\n")
            exit(-1)
else:
    IMAGE_SIZE = 0

if args.transform:
    try:
        DO_TRANSFORM = int(args.transform)
    except ValueError:
        print("\nInput Error type of transform option!\n")
        exit(-1)
    else:
        if DO_TRANSFORM != 1 and DO_TRANSFORM != 0:
            print("\nInput Error transform option!\n")
            exit(-1)


def show_options():
    pass
