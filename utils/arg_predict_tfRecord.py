from DMP.dataset.variables import columns_dict
from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\nUseAge : python predict_tfRecord.py -vector 'vector_file_name'\n\n")
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is tuning)"
                                                  "\nUseAge : python predict_tfRecord.py -model (tuning|cnn)\n\n",
                        default='tuning', type=str)
    parser.add_argument("-tensor_dir", "--tensor_dir", help="set directory name for log and tensor (default is Null)"
                                                            "\nUseAge : python predict_tfRecord.py -tensor_dir 'dir_name'\n\n")
    parser.add_argument("-save", "--save", help="save a score to csv file (default is '{vector}')"
                                                "\nUseAge : python predict_tfRecord.py -save 'NAME' (in analysis dir)\n\n")
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python predict_tfRecord.py -show 1 (True)\n\n")
    parser.add_argument("-plot", "--plot", help="set a option for visualization (default is 0)"
                                                "\nUseAge : python predict_tfRecord.py -plot 1 (True)\n\n")
    _args = parser.parse_args()

    return _args


args = get_arguments()

# LOAD options #
READ_VECTOR = False
vector_path = "modeling/vectors/"
vector_name = "vectors_dataset_parsing_5"

if args.vector:
    READ_VECTOR = args.vector
else:
    READ_VECTOR = vector_path + vector_name

# Training options #
TYPE_OF_MODEL = "svm"
TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR
type_of_features = [TYPE_OF_FEATURE] + [type_of_column for type_of_column in columns_dict]
IMAGE_PATH = str()

# Target options #
COLUMN_TARGET = False
COLUMN_TARGET_NAME = str()

# SHOW options #
DO_SHOW = False
DO_SHOW_PLOT = False

# SAVE options #
TENSOR_DIR_NAME = str()
SAVE_DIR_NAME = str()
DO_DELETE = False

EPOCH = int()
NUM_HIDDEN_LAYER = int()
LEARNING_RATE = float()


if args.model:
    TYPE_OF_MODEL = args.model
    if TYPE_OF_MODEL != "tuning" and TYPE_OF_MODEL != "cnn":
        print("\nInput Error model option! (You must input (svm|ffnn|cnn))\n")
        exit(-1)

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

if args.tensor_dir:
    TENSOR_DIR_NAME = args.tensor_dir

if TYPE_OF_FEATURE != KEY_NAME_OF_MERGE_VECTOR:
    TENSOR_DIR_NAME += "_" + TYPE_OF_FEATURE + "/"
else:
    TENSOR_DIR_NAME += "/"

if args.save:
    SAVE_DIR_NAME = args.save
else:
    SAVE_DIR_NAME = TENSOR_DIR_NAME

if args.plot:
    try:
        DO_SHOW_PLOT = int(args.plot)
    except ValueError:
        print("\nInput Error type of show option!\n")
        exit(-1)
    else:
        if DO_SHOW_PLOT != 1 and DO_SHOW_PLOT != 0:
            print("\nInput Error show option!\n")
            exit(-1)


def show_options():
    pass
