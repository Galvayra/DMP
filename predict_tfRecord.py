# -*- coding: utf-8 -*-
import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.learning.dataClassifier import DataClassifier
from DMP.learning.dataHandler import DataHandler


if __name__ == '__main__':
    classifier = DataClassifier(DataHandler())
    classifier.predict_tf_record()
