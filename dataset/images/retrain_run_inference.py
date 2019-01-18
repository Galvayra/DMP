# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import json

FLAGS = None

imagePath = 'dataset/test/'                                 # 추론을 진행할 이미지 경로
resultPath = 'dataset/result/'
alivePath = 'alive/'
deathPath = 'death/'
tpPath = 'tp/'
fpPath = 'fp/'
tnPath = 'tn/'
fnPath = 'fn/'
modelFullPath = 'save/output_graph.pb'                      # 읽어들일 graph 파일 경로
labelsFullPath = 'save/output_labels.txt'                   # 읽어들일 labels 파일 경로
logFullPath = 'save/inference.txt'

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


def get_images(path):
    if os.path.isdir(path):
        if os.path.isdir(path + alivePath) and os.path.isdir(path + deathPath):
            return [sorted(os.listdir(path + alivePath)), sorted(os.listdir(path + deathPath))]

    return False


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def inference_image(images, label_dir):
    global count_positive, count_negative, count_tp, count_fp, count_tn, count_fn
    global log_dict

    create_graph()
    with tf.Session() as sess:
        for image in images:
            image_data = tf.gfile.FastGFile(imagePath + label_dir + image, 'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            if predictions[0] > predictions[1]:
                count_positive += 1

                if label_dir == alivePath:
                    log_dict[fpPath].append(image)
                    count_fp += 1
                else:
                    log_dict[tpPath].append(image)
                    count_tp += 1
            else:
                count_negative += 1

                if label_dir == alivePath:
                    log_dict[tnPath].append(image)
                    count_tn += 1
                else:
                    log_dict[fnPath].append(image)
                    count_fn += 1


def run_inference_on_images(_):
    answer = None

    images = get_images(imagePath)

    if not images:
        tf.logging.fatal("File does not exist in %s", FLAGS.image_dir)
        return answer

    else:
        if len(images[0]) < 1:
            tf.logging.fatal('File does not exist %s', alivePath)
            return answer
        elif len(images[1]) < 1:
            tf.logging.fatal('File does not exist %s', deathPath)
            return answer

    inference_image(images[0], alivePath)
    inference_image(images[1], deathPath)

    print("\n\n\ncount positive -", count_positive)
    print("count negative -", count_negative)
    print("count tp -", count_tp)
    print("count fp -", count_fp)
    print("count tn -", count_tn)
    print("count fn -", count_fn)

    precision = float(count_tp) / (count_tp + count_fp)
    recall = float(count_tp) / (count_tp + count_fn)
    accuracy = float(count_tp + count_tn) / (count_tp + count_tn + count_fp + count_fn)

    print("Precision - %.2f" % (precision * 100))
    print("Recall    - %.2f" % (recall * 100))
    print("F1 score  - %.2f" % ((2 * ((precision * recall) / (precision + recall))) * 100))
    print("Accuracy  - %.2f" % (accuracy * 100))

    dump_log_dict()


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

    # make result directory
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run_inference_on_images, argv=[sys.argv[0]] + unparsed)
