import os
import numpy as np
from PIL import Image
import tensorflow as tf


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


def to_tfrecords(image_list, label_list, tfrecords_name):
    print("Start converting")
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(path=tfrecords_name, options=options)

    for image_path, label_path in zip(image_list, label_list):
        image = Image.open(image_path)
        label = Image.open(label_path)
        _binary_image = image.tostring()
        _binary_label = label.tostring()
        filename = os.path.basename(image_path)

        string_set = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'Image': _bytes_feature(_binary_image),
            'Label': _bytes_feature(_binary_label),
            'mean': _float_feature(image.mean().astype(np.float32)),
            'std': _float_feature(image.std().astype(np.float32)),
            'filename': _bytes_feature(str.encode(filename)),
        }))

        writer.write(string_set.SerializeToString())

    writer.close()
