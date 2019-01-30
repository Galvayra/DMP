import json
import shutil
import os
import argparse
import random

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-save_path", "--save_path", help="set a path of result"
                                                          "\n(Default is result)\n\n")
    parser.add_argument("-log_name", "--log_name", help="set a name of result inference file"
                                                        "\n(Default is 'inference.txt')\n\n")
    return parser.parse_args()


args = get_arguments()

savePath = str()
SAVE_DIR_NAME = 'result/'
TP_PATH = SAVE_DIR_NAME + 'tp/'
FP_PATH = SAVE_DIR_NAME + 'fp/'
TN_PATH = SAVE_DIR_NAME + 'tn/'
FN_PATH = SAVE_DIR_NAME + 'fn/'

log_name = 'inference.txt'

NUM_OF_COPY = 100

if args.save_path:
    savePath = args.save_path


def make_result_dir(path):
    if os.path.isdir(path + SAVE_DIR_NAME):
        shutil.rmtree(path + SAVE_DIR_NAME)
    os.mkdir(path + SAVE_DIR_NAME)
    os.mkdir(path + TP_PATH)
    os.mkdir(path + FP_PATH)
    os.mkdir(path + TN_PATH)
    os.mkdir(path + FN_PATH)


def read_log(_path):
    try:
        with open(_path, 'r') as r_file:
            return json.load(r_file)
    except FileNotFoundError:
        print("\nFile Not Found Error -", _path)
        print("\nPlease make sure load 'inference.txt'")
        exit(-1)


def copy_images(log_dict):
    for resultPath, result_dict in log_dict.items():
        # for img_name, src_path in result_dict.items():
        #     shutil.copyfile(src_path + img_name, savePath + SAVE_DIR_NAME + resultPath + img_name)

        for img_name in random.sample(result_dict.keys(), NUM_OF_COPY):
            src_path = result_dict[img_name]
            shutil.copyfile(src_path + img_name, savePath + SAVE_DIR_NAME + resultPath + img_name)


if __name__ == '__main__':
    make_result_dir(savePath)
    copy_images(read_log(savePath + log_name))
