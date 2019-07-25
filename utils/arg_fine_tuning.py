import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-train_dir", "--train_dir", help="set a image file directory for training"
                                                      "\nUseAge : python fine_tuning.py -train_dir 'T'\n\n")
parser.add_argument("-val_dir", "--val_dir", help="set a image file directory for validation"
                                                  "\nUseAge : python fine_tuning.py -val_dir 'V'\n\n")
parser.add_argument("-model_path", "--model_path", help="set a path for loading off-the-shelf model"
                                                        "\nUseAge : python fine_tuning.py -model_path 'M'\n\n",
                    default="learning/model/vgg_16.ckpt", type=str)
parser.add_argument('-show', '--show', default=True, type=bool)

parser.add_argument('--log', default='transfer', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

args = parser.parse_args()

VGG_MEAN = [123.68, 116.78, 103.94]

PATH_LOGS = "./logs/"
LOG_DIR_NAME = args.log
DO_SHOW = args.show
