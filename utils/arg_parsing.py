import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python parsing.py -input 'Name'\n\n")
    parser.add_argument("-output", "--output", help="set a name of output csv file"
                                                    "\ndefault is 'dataset_parsing'.csv "
                                                    "\nthe File will be saved in dataset dictionary"
                                                    "\nUseAge : python parsing.py -output 'Name'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\ndefault is 'None'.csv "
                                                    "\nUseAge : python parsing.py -target 'Symptom'\n\n")
    parser.add_argument("-sampling", "--sampling", help="set whether sampling or not (default is 0)"
                                                        "\nUseAge : python parsing.py -sampling 1\n\n")

    _args = parser.parse_args()

    return _args


args = get_arguments()

COLUMN_TARGET = False
COLUMN_TARGET_NAME = str()

READ_FILE = "dataset.csv"
SAVE_FILE = "dataset_parsing.csv"

DO_SAMPLING = False

if args.input:
    READ_FILE = args.input

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
        SAVE_FILE = SAVE_FILE.split(".csv")[0] + "_" + COLUMN_TARGET_NAME + ".csv"

if args.output:
    SAVE_FILE = args.output

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
