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
    _args = parser.parse_args()

    return _args


args = get_arguments()

if not args.input:
    READ_FILE = "dataset.csv"
else:
    READ_FILE = args.input

if not args.output:
    LOAD_FILE = "dataset_parsing.csv"
else:
    LOAD_FILE = args.output

if not args.target:
    COLUMN_TARGET = str()
else:
    COLUMN_TARGET = args.target
