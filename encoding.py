# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DeepPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DeepPredict.dataset.dataHandler import DataHandler
from DeepPredict.modeling.vectorization import MyVector
from DeepPredict.arguments import USE_ID


if __name__ == '__main__':

    if USE_ID.startswith("reverse#"):
        myData = MyVector(DataHandler(is_reverse=True))
    else:
        myData = MyVector(DataHandler())

    myData.encoding()
    myData.dump()
