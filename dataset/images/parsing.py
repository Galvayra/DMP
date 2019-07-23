# -*- coding: utf-8 -*-

from DMP.dataset.dataParser import DataParser
from DMP.dataset.images.variables import *


if __name__ == '__main__':
    dataParser = DataParser(IMAGE_CSV, ct_image_path=CT_IMAGE_PATH)
    dataParser.parsing()
    dataParser.save_log()
    dataParser.save()
