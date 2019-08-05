# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.dataset.dataParser import DataParser
from DMP.utils.arg_parsing import READ_FILE, DO_PARSING_IMAGE
from DMP.dataset.images.variables import *


if __name__ == '__main__':
    if DO_PARSING_IMAGE:
        dataParser = DataParser(IMAGE_CSV, ct_image_path=CT_IMAGE_PATH)
    else:
        dataParser = DataParser(READ_FILE)

    dataParser.parsing()
    dataParser.save()

    if DO_PARSING_IMAGE:
        dataParser.save_log()
