# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import json

try:
    import images
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from DMP.dataset.images.learning.score import show_scores

FLAGS = None

imagePath = 'dataset/test/'                                 # 추론을 진행할 이미지 경로
resultPath = 'dataset/result/'
alivePath = 'alive/'
deathPath = 'death/'
tpPath = 'tp/'
fpPath = 'fp/'
tnPath = 'tn/'
fnPath = 'fn/'
model_name = 'output_graph.pb'                      # 읽어들일 graph 파일 경로
labels_name = 'output_labels.txt'                   # 읽어들일 labels 파일 경로
save_name = 'inference.txt'

count_positive = int()
count_negative = int()
count_tp = int()
count_fp = int()
count_tn = int()
count_fn = int()

log_dict = {
    tpPath: list(),
    fpPath: list(),
    tnPath: list(),
    fnPath: list(),
}


def get_images(path, data_dict):
    image_dict = dict()

    if os.path.isdir(path):
        if os.path.isdir(path + alivePath) and os.path.isdir(path + deathPath):
            # append alive test image file
            for image_file_name in sorted(os.listdir(path + alivePath)):
                if image_file_name in data_dict[alivePath[:-1]]:
                    image_dict[image_file_name] = path + alivePath

            # append death test image file
            for image_file_name in sorted(os.listdir(path + deathPath)):
                if image_file_name in data_dict[deathPath[:-1]]:
                    image_dict[image_file_name] = path + deathPath

    return image_dict


def create_graph(model_path):
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def inference_image(model_path, image_dict):
    global count_positive, count_negative, count_tp, count_fp, count_tn, count_fn
    global log_dict

    y_test = list()
    y_prob = list()
    predictions = list()

    create_graph(model_path)
    with tf.Session() as sess:
        for image, path in image_dict.items():
            image_data = tf.gfile.FastGFile(path + image, 'rb').read()
            is_alive = path.endswith(alivePath)
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            prediction = list(np.squeeze(prediction))
            y_prob.append(prediction[-1])

            if is_alive:
                y_test.append(0)
            else:
                y_test.append(1)

            if prediction[0] > prediction[1]:
                predictions.append(0)
                count_positive += 1

                if is_alive:
                    log_dict[fpPath].append(image)
                    count_fp += 1
                else:
                    log_dict[tpPath].append(image)
                    count_tp += 1
            else:
                predictions.append(1)
                count_negative += 1

                if is_alive:
                    log_dict[tnPath].append(image)
                    count_tn += 1
                else:
                    log_dict[fnPath].append(image)
                    count_fn += 1

        show_scores(np.array(y_test), np.array(y_prob), np.array(predictions))


def set_new_paths(tensor_path):
    model_path = tensor_path + model_name
    labels_path = tensor_path + labels_name
    save_path = tensor_path + save_name

    return model_path, labels_path, save_path


def load_log(log_path):
    try:
        with open(log_path, 'r') as read_file:
            return json.load(read_file)
    except FileNotFoundError:
        return None


# def load_labels(labels_path):
#     try:
#         with open(labels_path, 'r') as read_file:
#             return [line.strip() for line in read_file]
#     except FileNotFoundError:
#         return None


def run_inference_on_images(_):
    model_path, labels_path, save_path = set_new_paths(FLAGS.tensor_path)
    data_dict = load_log(FLAGS.log_path)
    # labels = load_labels(labels_path)

    if not data_dict:
        print("\nThere is no log file for testing!\n")
        return -1

    # if not labels:
    #     print("\nThere is no labels for testing!\n")
    #     return -1

    image_dict = get_images(FLAGS.image_dir, data_dict["test"])
    if not image_dict:
        tf.logging.fatal("File does not exist in %s", FLAGS.image_dir)
        return -1

    inference_image(model_path, image_dict)

    #
    # print("\n\n\ncount positive -", count_positive)
    # print("count negative -", count_negative)
    # print("count tp -", count_tp)
    # print("count fp -", count_fp)
    # print("count tn -", count_tn)
    # print("count fn -", count_fn)
    #
    # precision = float(count_tp) / (count_tp + count_fp)
    # recall = float(count_tp) / (count_tp + count_fn)
    # accuracy = float(count_tp + count_tn) / (count_tp + count_tn + count_fp + count_fn)
    #
    # print("Precision - %.2f" % (precision * 100))
    # print("Recall    - %.2f" % (recall * 100))
    # print("F1 score  - %.2f" % ((2 * ((precision * recall) / (precision + recall))) * 100))
    # print("Accuracy  - %.2f" % (accuracy * 100))

    # dump_log_dict()


def dump_log_dict():
    with open(logFullPath, 'w') as outfile:
        json.dump(log_dict, outfile, indent=4)
        print("\n=========================================================")
        print("\nsuccess make dump file! - file name is", logFullPath, "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default='',
        help='Path to log file which has information of train, validation, test rate.'
    )
    parser.add_argument(
        '--tensor_path',
        type=str,
        default='',
        help='Path to load tensor directory name.'
    )

    # make result directory
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run_inference_on_images, argv=[sys.argv[0]] + unparsed)
