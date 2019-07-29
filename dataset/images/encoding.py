# -*- coding: utf-8 -*-

from DMP.dataset.images.variables import CT_IMAGE_PATH
from DMP.modeling.vectorMaker import VectorMaker
from DMP.modeling.variables import KEY_TOTAL, KEY_TRAIN, KEY_VALID, KEY_TEST
from DMP.dataset.dataHandler import DataHandler
from DMP.utils.arg_encoding import *


if __name__ == '__main__':
    file_dict = {
        KEY_TOTAL: SAVE_FILE_TOTAL,
        KEY_TRAIN: SAVE_FILE_TRAIN,
        KEY_VALID: SAVE_FILE_VALID,
        KEY_TEST: SAVE_FILE_TEST
    }
    dataHandler_dict = dict()

    # loading data
    for key, read_csv in file_dict.items():
        dataHandler = DataHandler(read_csv, column_target=COLUMN_TARGET, eliminate_target=True)
        dataHandler.load()
        dataHandler_dict[key] = dataHandler

    # encoding data using dataHandler
    vectorMaker = VectorMaker(dataHandler_dict, ct_image_path=CT_IMAGE_PATH)
    vectorMaker.encoding()
    vectorMaker.dump()
