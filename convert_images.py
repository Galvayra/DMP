# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.dataset.images.imageConverter import ImageConverter

if __name__ == '__main__':
    imageConverter = ImageConverter()
    imageConverter.convert()
    imageConverter.save_log()
