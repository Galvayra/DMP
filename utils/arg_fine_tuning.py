from DMP.modeling.variables import KEY_NAME_OF_MERGE_VECTOR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-vector", "--vector", help="set vector file name to train or predict"
                                                    "\n(default is 'image_vector')"
                                                    "\nUseAge : python fine_tuning.py -vector 'vector_file_name'\n\n",
                        default="modeling/vectors/image_vector", type=str)
    parser.add_argument("-model", "--model", help="set a model type of neural net (default is tuning)"
                                                  "\nUseAge : python fine_tuning.py -model (tuning|cnn)\n\n",
                        default="tuning", type=str)
    parser.add_argument("-epoch", "--epoch", help="set epoch for neural network (default is 2000)"
                                                  "\nyou have to use this option more than 100"
                                                  "\nUseAge : python fine_tuning.py -epoch 2000\n\n",
                        default=2000, type=int)
    parser.add_argument("-hidden", "--hidden", help="set a number of hidden layer (default is 0)"
                                                    "\ndefault is not using hidden layer for linear model"
                                                    "\nUseAge : python fine_tuning.py -hidden 2 (non-linear)\n\n",
                        default=0, type=int)
    parser.add_argument("-learn", "--learn", help="set a learning rate for training (default is 0.0001)"
                                                  "\nUseAge : python fine_tuning.py -learn 0.01\n\n",
                        default=0.0001, type=float)
    parser.add_argument("-tensor_dir", "--tensor_dir", help="set directory name for log and tensor"
                                                            "\nUseAge : python fine_tuning.py -tensor_dir 'dir'\n\n",
                        default="image_vector", type=str)
    parser.add_argument("-delete", "--delete", help="set whether SAVE_DIR will be delete (default is 1)"
                                                    "\nIf you already have dir for saving, delete it and then save"
                                                    "\nIf you set False, It will be stopped before training"
                                                    "\nUseAge : python fine_tuning.py -delete 1 (True)\n\n",
                        default=True, type=bool)
    parser.add_argument("-show", "--show", help="show score of mortality and immortality (default is 0)"
                                                "\nUseAge : python fine_tuning.py -show 1 (True)\n\n",
                        default=False, type=bool)
    parser.add_argument("-ver", "--version", help="set a version for training (Default is 1)"
                                                  "\n1 - training ct images respectively"
                                                  "\n2 - training ct images recursively"
                                                  "\nUseAge : python fine_tuning.py -ver 1\n\n",
                        default=1, type=int)
    _args = parser.parse_args()

    return _args


args = get_arguments()

VERSION = args.version

if VERSION != 1 and VERSION != 2:
    print("\nInput Error Boundary of version option!\n")
    exit(-1)

# Path of vector #
READ_VECTOR = args.vector

# Model option #
TYPE_OF_MODEL = args.model

if TYPE_OF_MODEL != "tuning" and TYPE_OF_MODEL != "cnn":
    print("\nInput Error model option! (You must input - ['tuning', 'cnn'])\n")
    exit(-1)

# Target options #
COLUMN_TARGET = False
COLUMN_TARGET_NAME = str()

# Parameter options #
EPOCH = args.epoch
NUM_HIDDEN_LAYER = args.hidden
NUM_HIDDEN_DIMENSION = 0
LEARNING_RATE = args.learn

TYPE_OF_FEATURE = KEY_NAME_OF_MERGE_VECTOR

# SHOW options #
DO_SHOW = args.show

if DO_SHOW != 1 and DO_SHOW != 0:
    print("\nInput Error show option!\n")
    exit(-1)

# SAVE options #
TENSOR_DIR_NAME = args.tensor_dir + "/"
DO_DELETE = args.delete

if DO_DELETE != 1 and DO_DELETE != 0:
    print("\nInput Error delete option!\n")
    exit(-1)


def show_options():
    pass

# parser.add_argument("-train_dir", "--train_dir", help="set a image file directory for training"
#                                                       "\nUseAge : python fine_tuning.py -train_dir 'T'\n\n")
# parser.add_argument("-val_dir", "--val_dir", help="set a image file directory for validation"
#                                                   "\nUseAge : python fine_tuning.py -val_dir 'V'\n\n")
# parser.add_argument("-model_path", "--model_path", help="set a path for loading off-the-shelf model"
#                                                         "\nUseAge : python fine_tuning.py -model_path 'M'\n\n",
#                     default="learning/model/vgg_16.ckpt", type=str)
# parser.add_argument('-show', '--show', default=True, type=bool)
#
# parser.add_argument('--log', default='transfer', type=str)
# parser.add_argument('--batch_size', default=32, type=int)
# parser.add_argument('--num_workers', default=4, type=int)
# parser.add_argument('--num_epochs1', default=10, type=int)
# parser.add_argument('--num_epochs2', default=10, type=int)
# parser.add_argument('--learning_rate1', default=1e-3, type=float)
# parser.add_argument('--learning_rate2', default=1e-5, type=float)
# parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
# parser.add_argument('--weight_decay', default=5e-4, type=float)
#
# args = parser.parse_args()
#
# VGG_MEAN = [123.68, 116.78, 103.94]
#
# PATH_LOGS = "./logs/"
# LOG_DIR_NAME = args.log
# DO_SHOW = args.show
