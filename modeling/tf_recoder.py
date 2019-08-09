import cv2
import tensorflow as tf
import numpy as np
from .variables import EXTENSION_OF_IMAGE
from DMP.learning.variables import IMAGE_RESIZE

EXTENSION_OF_TF_RECORD = ".tfrecords"


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
    tf_record_path += EXTENSION_OF_TF_RECORD
    with tf.python_io.TFRecordWriter(tf_record_path) as writer:
        for i in range(len(image_list)):
            img_path = image_list[i]
            img = load_image(img_path)
            label = label_list[i]

            # Create a feature
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(tf.compat.as_bytes(img.tobytes()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        print('success converting to the tfrecords -', tf_record_path)


def get_img_from_tf_records(tf_record_path):
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tf_record_path], num_epochs=1)
    # print(filename_queue)
    # Define a reader and read the next record

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [IMAGE_RESIZE, IMAGE_RESIZE, 3])

    # Cast label data into int32
    label = tf.cast(features['label'], tf.int32)

    print('success read to the tfrecords -', tf_record_path)

    return image, label


def get_tf_record_path(img_path):
    image_path_list = img_path.split('/')
    folder_name = image_path_list[-2]
    file_name = image_path_list[-1].split(EXTENSION_OF_IMAGE)[0]

    return folder_name + "_" + file_name + EXTENSION_OF_TF_RECORD


def load_image(img_path):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img


# # Load the image
# def __get_img(img_path):
#     return Image.open(img_path)
