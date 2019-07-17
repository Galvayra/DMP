# -*- coding: utf-8 -*-

from DMP.dataset.dataParser import DataParser
import DMP.utils.arg_parsing

IMAGE_CSV = "dataset_images.csv"


if __name__ == '__main__':
    dataParser = DataParser(IMAGE_CSV)
    dataParser.parsing()
    dataParser.save()
