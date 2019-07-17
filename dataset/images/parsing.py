# -*- coding: utf-8 -*-

from DMP.dataset.dataParser import DataParser
import DMP.utils.arg_parsing

IMAGE_CSV = "dataset_images.csv"
IMAGE_PATH = "ct_images/"


if __name__ == '__main__':
    dataParser = DataParser(IMAGE_CSV, ct_image_path=IMAGE_PATH)
    dataParser.parsing()
    dataParser.save()
