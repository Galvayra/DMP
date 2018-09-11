# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.dataset.dataHandler import DataHandler
from DMP.modeling.vectorization import MyVector
from DMP.arguments import USE_ID


if __name__ == '__main__':

    if USE_ID.startswith("reverse#"):
        myData = MyVector(DataHandler(is_reverse=True))
    else:
        myData = MyVector(DataHandler())

    # myData.encoding()
    # myData.dump()
