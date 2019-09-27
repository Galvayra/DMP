
import sys
from os import path

try:
    import DMP
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DMP.learning.slimLearner import SlimLearner


if __name__ == '__main__':
    nn = SlimLearner()
    nn.show_tf_record_info()

