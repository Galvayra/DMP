# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.modeling.vectorMaker import VectorMaker
from DMP.dataset.dataHandler import DataHandler
from DMP.utils.arg_encoding import *


if __name__ == '__main__':
    if USE_ID.startswith("reverse#"):
        dataHandler = DataHandler(READ_FILE, is_reverse=True)
    else:
        dataHandler = DataHandler(READ_FILE)

    # loading data
    dataHandler.load()

    # encoding data using dataHandler
    vectorMaker = VectorMaker(dataHandler)
    vectorMaker.encoding()
    vectorMaker.dump()
