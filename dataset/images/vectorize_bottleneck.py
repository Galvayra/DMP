import json
import shutil
import os
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def get_arguments():
    parser.add_argument("-save_path", "--save_path", help="set a path of result\n"
                                                          "(Default is result/)\n\n")
    parser.add_argument("-log_path", "--log_path", help="set a name of log file\n"
                                                        "(Default is None)\n\n")
    parser.add_argument("-output", "--output", help="set a name of vector\n"
                                                    "(Default is '../../modeling/vectors/{$save_path}_bottleneck')\n\n")
    return parser.parse_args()


arg = get_arguments()

TYPE_OF_FEATURE = 'merge'
BOTTLENECK_PATH = 'bottleneck/'
alivePath = 'alive/'
deathPath = 'death/'

vector_matrix = OrderedDict()
vector_matrix = {
    "feature": dict(),
    "x_train": {
        TYPE_OF_FEATURE: list()
    },
    "y_train": list(),
    "x_valid": {
        TYPE_OF_FEATURE: list()
    },
    "y_valid": list(),
    "x_test": {
        TYPE_OF_FEATURE: list()
    },
    "y_test": list()
}

savePath = 'result/' + BOTTLENECK_PATH

if arg.save_path:
    savePath = arg.save_path + BOTTLENECK_PATH

if arg.log_path:
    try:
        with open(arg.log_path, 'r') as read_file:
            load_dict = json.load(read_file)
    except FileNotFoundError:
        print("Please Input log_path option!!")
        exit(-1)

if arg.output:
    SAVE_VECTOR = arg.output
else:
    SAVE_VECTOR = '../../modeling/vectors/' + savePath.split('/')[1] + '_bottleneck'


def vectorization():
    vectorize(key="train")
    vectorize(key="valid")
    vectorize(key="test")


def vectorize(key):
    alive_bottleneck_list = get_bottleneck_list(key, alivePath[:-1])
    death_bottleneck_list = get_bottleneck_list(key, deathPath[:-1])

    for i in range(1, len(alive_bottleneck_list) + len(death_bottleneck_list) + 1):
        bottleneck = key + '_' + str(i) + '.jpg.txt'

        # append y label
        if bottleneck in alive_bottleneck_list:
            path_key = alivePath[:-1]
            vector_matrix["y_" + key].append([float(0)])
        else:
            path_key = deathPath[:-1]
            vector_matrix["y_" + key].append([float(1)])

        # append x data (vector)
        vector_matrix["x_" + key][TYPE_OF_FEATURE].append(get_vector(savePath + path_key + '/' + bottleneck))


def get_bottleneck_list(i, j):
    return [image + '.txt' for image in load_dict[i][j]]


def get_vector(bottleneck_path):
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    try:
        vector = [float(x) for x in bottleneck_string.split(',')]
        val_min = min(vector)
        val_diff = max(vector) - val_min

        return [(x - val_min) / val_diff for x in vector]
    except ValueError:
        print('Invalid float found, recreating bottleneck -', bottleneck_path)
        exit(-1)


def dump_vector():
    def __counting_mortality(_data):
        count = 0
        for _d in _data:
            if _d == [1]:
                count += 1

        return count

    with open(SAVE_VECTOR, 'w') as outfile:
        json.dump(vector_matrix, outfile, indent=4)
        print("\n\nsuccess make dump file! - file name is", SAVE_VECTOR)

    print("\nTrain total count -", str(len(vector_matrix["x_train"][TYPE_OF_FEATURE])).rjust(4),
          "\tmortality count -", str(__counting_mortality(vector_matrix["y_train"])).rjust(4))
    print("Valid total count -", str(len(vector_matrix["x_valid"][TYPE_OF_FEATURE])).rjust(4),
          "\tmortality count -", str(__counting_mortality(vector_matrix["y_valid"])).rjust(4))
    print("Test  total count -", str(len(vector_matrix["x_test"][TYPE_OF_FEATURE])).rjust(4),
          "\tmortality count -", str(__counting_mortality(vector_matrix["y_test"])).rjust(4), "\n\n")


if __name__ == '__main__':
    vectorization()
    dump_vector()
