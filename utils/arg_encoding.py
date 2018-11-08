import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset_parsing'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python encoding.py -input 'Name'\n\n")
    parser.add_argument("-output", "--output", help="set vector file name to train or predict (default is 'model')"
                                                    "\nUseAge : python encoding.py -output 'NAME_OF_VECTOR'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\nUseAge : python encoding.py -target 'symptom'\n\n")
    parser.add_argument("-w2v", "--w2v", help="using word2vec or not (default is 1)"
                                              "\nUseAge : python encoding.py -w2v 1 (True)\n\n")
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

USE_W2V = True

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
