import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset_parsing'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python encoding.py -input 'Name'\n\n")
    parser.add_argument("-target", "--target", help="set a target of specific symptom "
                                                    "\ndefault is 'None'.csv "
                                                    "\nUseAge : python encoding.py -target 'Symptom'\n\n")
    parser.add_argument("-ratio", "--ratio", help="set a ratio of training set (default is 8)"
                                                  "\ntrain:test:validate = N:(10-N)/2:(10-N)/2"
                                                  "\nUseAge : python encoding.py -ratio 6 (6:2:2)\n\n")
    parser.add_argument("-fold", "--fold", help="set a number of k-fold (default is 5)"
                                                "\nUseAge : python encoding.py -fold 5\n\n")
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python encoding.py -closed 1\n\n")
    parser.add_argument("-id", "--identify", help="set id for separating training sets (default is None)"
                                                  "\nUseAge : python encoding.py -id string\n\n")
    # parser.add_argument("-w2v", "--word2v", help="using word2vec (default is 0)"
    #                                              "\nUseAge : python training.py -w2v 1 (True)"
    #                                              "\n         python training.py -w2v 0 (False)\n\n")
    parser.add_argument("-output", "--output", help="set vector file name to train or predict (default is Null)"
                                                    "\nUseAge : python encoding.py -output 'file_name'\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

RATIO = 8
NUM_FOLDS = 5
IS_CLOSED = False
USE_W2V = False

READ_FILE = "dataset_parsing.csv"
COLUMN_TARGET = str()
USE_ID = str()
FILE_VECTOR = str()

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

if args.input:
    READ_FILE = args.input

if args.fold:
    try:
        NUM_FOLDS = int(args.fold)
    except ValueError:
        print("\nInput Error type of fold option!\n")
        exit(-1)
    else:
        if NUM_FOLDS < 1 or NUM_FOLDS > 10:
            print("\nInput Error a boundary of fold option!\n")
            exit(-1)

if args.ratio:
    try:
        RATIO = int(args.ratio)
    except ValueError:
        print("\nInput Error type of ratio option!\n")
        exit(-1)
    else:
        if RATIO < 1 or RATIO > 10:
            print("\nInput Error a boundary of ratio option!\n")
            exit(-1)

if args.closed:
    try:
        closed = int(args.closed)
    except ValueError:
        print("\nInput Error type of closed option!\n")
        exit(-1)
    else:
        if closed != 1 and closed != 0:
            print("\nInput Error closed option!\n")
            exit(-1)
        if closed == 1:
            IS_CLOSED = True
        else:
            IS_CLOSED = False

if args.identify:
    USE_ID = args.identify + "#"

if args.output:
    FILE_VECTOR = args.output + "_" + str(RATIO)

# if not args.word2v:
#     USE_W2V = False
# else:
#     try:
#         USE_W2V = int(args.word2v)
#     except ValueError:
#         print("\nInput Error type of word2v option!\n")
#         exit(-1)
#     else:
#         if USE_W2V != 1 and USE_W2V != 0:
#             print("\nInput Error word2v option!\n")
#             exit(-1)
