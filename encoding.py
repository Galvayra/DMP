# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.modeling.vectorMaker import VectorMaker
from DMP.dataset.dataHandler import DataHandler
from DMP.utils.arg_encoding import READ_FILE, COLUMN_TARGET


if __name__ == '__main__':
    # loading data
    dataHandler = DataHandler(READ_FILE, column_target=COLUMN_TARGET, eliminate_target=True)
    dataHandler.load()

    # encoding data using dataHandler
    vectorMaker = VectorMaker(dataHandler)
    vectorMaker.encoding()
    vectorMaker.dump()
