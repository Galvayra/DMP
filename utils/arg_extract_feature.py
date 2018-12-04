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
    parser.add_argument("-ntree", "--ntree", help="set a number of tree in random forest (default is 400)"
                                                  "\nUseAge : python extract_feature.py -ntree 400\n\n")
    parser.add_argument("-show", "--show", help="show importance features (default is 0)"
                                                "\nUseAge : python extract_feature.py -show 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# LOAD options #
READ_VECTOR = False
vector_path = "modeling/vectors/"
vector_name = "model"

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR

DO_SHOW = False

# Random Forest Option #
NUM_OF_TREE = 400


if args.vector:
    READ_VECTOR = args.vector
else:
    READ_VECTOR = vector_path + vector_name

if args.output:
    SAVE_VECTOR = args.output
else:
    SAVE_VECTOR = READ_VECTOR + "_new"

if args.show:
    try:
        DO_SHOW = int(args.show)
    except ValueError:
        print("\nInput Error type of show option!\n")
        exit(-1)
    else:
        if DO_SHOW != 1 and DO_SHOW != 0:
            print("\nInput Error show option!\n")
            exit(-1)

if args.ntree:
    try:
        NUM_OF_TREE = int(args.ntree)
    except ValueError:
        print("\nInput Error type of ntree option!\n")
        exit(-1)
    else:
        if NUM_OF_TREE < 10 or NUM_OF_TREE > 10000:
            print("\nInput Error ntree option!\n")
            exit(-1)
