# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.dataset.images.imageSplitter import ImageSplitter


if __name__ == '__main__':
    imageSplitter = ImageSplitter()
    imageSplitter.save_ct_dict2log()
