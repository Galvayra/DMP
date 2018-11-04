import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset_parsing'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python encoding.py -input 'Name'\n\n")
    parser.add_argument("-output", "--output", help="set vector file name to train or predict (default is Null)"
                                                    "\nUseAge : python encoding.py -output 'file_name'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\ndefault is 'None'.csv "
                                                    "\nUseAge : python encoding.py -target 'Symptom'\n\n")
    parser.add_argument("-ratio", "--ratio", help="set a ratio of training set in data set(default is 0.8)"
                                                  "\ntrain:test:validate = N:(10-N)/2:(10-N)/2"
                                                  "\nUseAge : python encoding.py -ratio 0.5 (2:1:1)\n\n")
    parser.add_argument("-w2v", "--w2v", help="using word2vec or not"
                                              "\nUseAge : python encoding.py -id string\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

READ_FILE = "dataset_parsing.csv"
COLUMN_TARGET = str()
FILE_VECTOR = str()

RATIO = 0.8
USE_W2V = True

if args.input:
    READ_FILE = args.input

if args.output:
    FILE_VECTOR = args.output

if args.target:
    COLUMN_TARGET = args.target

    if COLUMN_TARGET == "b":
        COLUMN_TARGET = "CR"
        READ_FILE = READ_FILE.split(".csv")[0] + "_bacteremia" + ".csv"
    elif COLUMN_TARGET == "s":
        COLUMN_TARGET = "CU"
        READ_FILE = READ_FILE.split(".csv")[0] + "_sepsis" + ".csv"
    elif COLUMN_TARGET == "p":
        COLUMN_TARGET = "CS"
        READ_FILE = READ_FILE.split(".csv")[0] + "_pneumonia" + ".csv"

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
