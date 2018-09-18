# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.modeling.vectorMaker import VectorMaker
from DMP.dataset.dataParser import DataParser
from DMP.dataset.variables import LOAD_FILE
from DMP.utils.arguments import USE_ID


if __name__ == '__main__':
    if USE_ID.startswith("reverse#"):
        dataParser = DataParser(LOAD_FILE, is_reverse=True)
    else:
        dataParser = DataParser(LOAD_FILE)

    # parsing data
    dataParser.parsing()

    # encoding data using dataParser
    vectorMaker = VectorMaker(dataParser)
    # vectorMaker.encoding()
    # vectorMaker.dump()
