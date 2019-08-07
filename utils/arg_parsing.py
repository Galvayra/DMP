import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-base", "--base",
                        help="set a path of log file name for restore(default is None)"
                             "\nUseAge : python encoding.py -base 'B'\n\n",
                        default="", type=str)
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python parsing.py -input 'Name'\n\n")
    parser.add_argument("-output", "--output", help="set a name of output csv file"
                                                    "\ndefault is 'parsing'.csv "
                                                    "\nthe File will be saved in dataset/parsing"
                                                    "\nUseAge : python parsing.py -output 'Name'\n\n")
    parser.add_argument("-ratio", "--ratio", help="set a ratio of training set in data set(default is 0.8)"
                                                  "\ntrain:test:validate = N:(10-N)/2:(10-N)/2"
                                                  "\nUseAge : python encoding.py -ratio 0.5 (2:1:1)\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\nUseAge : python parsing.py -target 'symptom'\n\n")
    parser.add_argument("-sampling", "--sampling", help="set whether sampling or not (default is 1)"
                                                        "\nUseAge : python parsing.py -sampling 1\n\n")
    parser.add_argument("-parsing_image", "--parsing_image",
                        help="set whether removing which doesn't have ct images or not (default is 0)"
                             "\nUseAge : python parsing.py -parsing_image 1\n\n",
                        default=0, type=int)
    _args = parser.parse_args()

    return _args


args = get_arguments()

COLUMN_TARGET = False
COLUMN_TARGET_NAME = str()

EXTENSION_FILE = ".csv"
READ_FILE = "dataset" + EXTENSION_FILE
SAVE_FILE_TOTAL = "parsing"
SAVE_FILE_TRAIN = "parsing_train"
SAVE_FILE_VALID = "parsing_valid"
SAVE_FILE_TEST = "parsing_test"

DO_SAMPLING = True
DO_PARSING_IMAGE = args.parsing_image

if DO_PARSING_IMAGE != 1 and DO_PARSING_IMAGE != 0:
    print("\nInput Error parsing_image option!\n")
    exit(-1)

LOG_NAME = args.base
RATIO = 0.8

if args.input:
    READ_FILE = args.input + EXTENSION_FILE

if args.output:
    SAVE_FILE_TOTAL = args.output
    SAVE_FILE_TRAIN = args.output + "_train"
    SAVE_FILE_VALID = args.output + "_valid"
    SAVE_FILE_TEST = args.output + "_test"

if args.target:
    COLUMN_TARGET = args.target
    COLUMN_TARGET_NAME = str()

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
        COLUMN_TARGET = False

    if COLUMN_TARGET:
        SAVE_FILE_TRAIN = SAVE_FILE_TOTAL + "_" + COLUMN_TARGET_NAME + "_train" + EXTENSION_FILE
        SAVE_FILE_VALID = SAVE_FILE_TOTAL + "_" + COLUMN_TARGET_NAME + "_valid" + EXTENSION_FILE
        SAVE_FILE_TEST = SAVE_FILE_TOTAL + "_" + COLUMN_TARGET_NAME + "_test" + EXTENSION_FILE
        SAVE_FILE_TOTAL = SAVE_FILE_TOTAL + "_" + COLUMN_TARGET_NAME + EXTENSION_FILE
else:
    SAVE_FILE_TOTAL += EXTENSION_FILE
    SAVE_FILE_TRAIN += EXTENSION_FILE
    SAVE_FILE_VALID += EXTENSION_FILE
    SAVE_FILE_TEST += EXTENSION_FILE

if args.ratio:
    try:
        RATIO = float(args.ratio)
    except ValueError:
        print("\nInput Error type of ratio option!\n")
        exit(-1)
    else:
        if RATIO < 0 or RATIO > 1:
            print("\nInput Error a boundary of ratio option!\n")
            exit(-1)

if args.sampling:
    try:
        sampling = int(args.sampling)
    except ValueError:
        print("\nValue Error of sampling option!\n")
        exit(-1)
    else:
        if sampling != 1 and sampling != 0:
            print("\nBoundary Error of sampling option!\n")
            exit(-1)

        if sampling == 1:
            DO_SAMPLING = True
        else:
            DO_SAMPLING = False
