import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-input", "--input", help="set a name of input csv file"
                                                  "\ndefault is 'dataset_parsing'.csv"
                                                  "\nthe File will be loaded in dataset dictionary"
                                                  "\nUseAge : python parsing.py -input 'Name'\n\n")
    parser.add_argument("-fold", "--fold", help="set a number of k-fold (default is 5)"
                                                "\nUseAge : python encoding.py -fold 5\n\n")
    parser.add_argument("-closed", "--closed", help="set closed or open data (default is 0)"
                                                    "\nUseAge : python encoding.py -closed 1\n\n")
    parser.add_argument("-id", "--identify", help="set id for separating training sets (default is None)"
                                                  "\nUseAge : python encoding.py -id string\n\n")
    parser.add_argument("-w2v", "--word2v", help="using word2vec (default is 0)"
                                                 "\nUseAge : python training.py -w2v 1 (True)"
                                                 "\n         python training.py -w2v 0 (False)\n\n")
    parser.add_argument("-train", "--train", help="set vector file name to train or predict (default is Null)"
                                                  "\nUseAge : python training.py -train 'file_name'\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

RATIO = 10


if not args.input:
    READ_FILE = "dataset_parsing.csv"
else:
    READ_FILE = args.input

if not args.fold:
    NUM_FOLDS = 5
else:
    try:
        NUM_FOLDS = int(args.fold)
    except ValueError:
        print("\nInput Error type of fold option!\n")
        exit(-1)
    else:
        if NUM_FOLDS < 1 or NUM_FOLDS > 10:
            print("\nInput Error a boundary of fold option!\n")
            exit(-1)

if not args.closed:
    IS_CLOSED = False
else:
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

if not args.identify:
    USE_ID = str()
else:
    USE_ID = args.identify + "#"

if not args.word2v:
    USE_W2V = False
else:
    try:
        USE_W2V = int(args.word2v)
    except ValueError:
        print("\nInput Error type of word2v option!\n")
        exit(-1)
    else:
        if USE_W2V != 1 and USE_W2V != 0:
            print("\nInput Error word2v option!\n")
            exit(-1)

if not args.train:
    FILE_VECTOR = str()
else:
    FILE_VECTOR = args.train
