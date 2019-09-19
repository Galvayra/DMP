# dump option for vector
DUMP_PATH = "vectors/"
DUMP_FILE = "vectors"

# key for handlers
KEY_TOTAL = "total"
KEY_TRAIN = "train"
KEY_VALID = "valid"
KEY_TEST = "test"
KEY_IMG_TRAIN = "img_train"
KEY_IMG_VALID = "img_valid"
KEY_IMG_TEST = "img_test"
KEY_TF_NAME = "tf_name"

# word embedding option
W2V_PATH = "embedding/"
W2V_FILE = "model.txt"

MODELING_PATH = "modeling/"
TF_RECORD_PATH = "tf_records/"
EXTENSION_OF_IMAGE = '.jpg'
# NAME_OF_TF_RECORD = "images"

KEY_NAME_OF_MERGE_VECTOR = "merge"

DIMENSION_W2V = 300
# Do not use  "[0.0, 0.0]" !!
# SCALAR_VECTOR = [0.0, 0.0]
SCALAR_VECTOR = [0.0]
SCALAR_DEFAULT_WEIGHT = 0.1
USE_STANDARD_SCALE = False
EXTENDED_WORD_VECTOR = True

USE_QUANTIZATION = True
USE_CLASS_VECTOR = False
NUM_QUANTIZE = 20
