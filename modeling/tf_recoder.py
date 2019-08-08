from PIL import Image
import tensorflow as tf
from .variables import EXTENSION_OF_IMAGE, EXTENSION_OF_TF_RECORD


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""
    if isinstance(text, str):
        return text
    elif isinstance(text, 'unicode'):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)


def to_tf_records(image_list, label_list, tf_record_path):
    for i in range(len(image_list)):
        img_path = image_list[i]
        save_tf_record_path = tf_record_path + get_tf_record_path(img_path)

        with tf.python_io.TFRecordWriter(save_tf_record_path) as writer:
            img = __get_img(img_path)
            label = label_list[i][0]

            # Create a feature
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(tf.compat.as_bytes(img.tobytes()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())


def get_tf_record_path(img_path):
    image_path_list = img_path.split('/')
    folder_name = image_path_list[-2]
    file_name = image_path_list[-1].split(EXTENSION_OF_IMAGE)[0]

    return folder_name + "_" + file_name + EXTENSION_OF_TF_RECORD


# Load the image
def __get_img(img_path):
    return Image.open(img_path)
