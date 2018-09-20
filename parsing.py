# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.dataset.dataParser import DataParser
from DMP.utils.arg_parsing import READ_FILE


if __name__ == '__main__':
    dataParser = DataParser(READ_FILE)
    dataParser.parsing()
    # dataParser.save()
