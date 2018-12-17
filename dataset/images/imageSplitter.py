# -*- coding: utf-8 -*-
from .variables import *
from DMP.dataset.dataHandler import DataHandler
import os
import shutil


class ImageSplitter(DataHandler):
    def __init__(self):
        super().__init__(read_csv=READ_CSV, data_path=DATA_PATH, do_what="splitting")
        self.count()
        self.make_train_test_dir()

    @staticmethod
    def make_train_test_dir():
        def __make_dir__(_path):
            if os.path.isdir(_path):
                shutil.rmtree(_path)
            os.mkdir(_path)

            print("\nSuccess to make directory -", _path)

        __make_dir__(DATA_PATH + TRAIN_DIR)
        __make_dir__(DATA_PATH + TEST_DIR)

    def count(self):
        cnt_mortality = self.counting_mortality(self.y_data)
        cnt_total = len(self.y_data)
        print("\n# of     total -", str(cnt_total).rjust(4),
              "\n# of     alive -", str(cnt_total - cnt_mortality).rjust(4),
              "\n# of mortality -", str(cnt_mortality).rjust(4), "\n\n")
