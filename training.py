# -*- coding: utf-8 -*-
import sys
import json
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.learning.train import MyTrain
from DMP.utils.arg_training import READ_VECTOR, show_options


if __name__ == '__main__':

    # if op.FILE_VECTOR:
    #     file_name = op.FILE_VECTOR
    # else:
    #     if op.USE_W2V:
    #         append_name = "w2v_"
    #     else:
    #         append_name = ""
    #
    #     if op.USE_ID:
    #         append_name += op.USE_ID
    #
    #     if op.IS_CLOSED:
    #         append_name += "closed_"
    #
    #     file_name = DUMP_FILE + "_" + append_name + csv_name + "_" + str(op.NUM_FOLDS)
    #     print(file_name)
    try:
        with open(READ_VECTOR, 'r') as file:
            vector_list = json.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
        print("FileNotFoundError] READ_VECTOR is", "'" + READ_VECTOR + "'", "\n\n")
    else:
        print("\nRead vectors -", READ_VECTOR)
        show_options()

        # for vector_dict in vector_list:
        #     for k in vector_dict:
        #         if type(vector_dict[k]) is dict:
        #             for j in vector_dict[k]:
        #                 print(j)

        train = MyTrain(vector_list)
        train.training()
