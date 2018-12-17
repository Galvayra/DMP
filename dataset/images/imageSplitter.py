# -*- coding: utf-8 -*-
from .variables import DATA_PATH, READ_CSV
from DMP.dataset.dataHandler import DataHandler


class ImageSplitter(DataHandler):
    def __init__(self):
        super().__init__(read_csv=READ_CSV, data_path=DATA_PATH, do_what="splitting")
