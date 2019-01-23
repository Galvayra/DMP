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
check_name = 'model-'

count_positive = int()
count_negative = int()
count_tp = int()
count_fp = int()
count_tn = int()
count_fn = int()

log_dict = {
    tpPath: dict(),
    fpPath: dict(),
    tnPath: dict(),
    fnPath: dict(),
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


def create_graph(model_path, chk_point_path, load_step):
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    if not load_step:
        # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        return False, False
    else:
        checkpoint = tf.train.get_checkpoint_state(chk_point_path)
        paths = checkpoint.all_model_checkpoint_paths
        index = [path.split(check_name)[-1] for path in paths].index(load_step)

        if index >= 0:
            print("\n\nCheck Point -", check_name + load_step, "\n\n")
            return checkpoint, paths[index]
        else:
            return False, False


def inference_image(model_path, chk_point_path, image_dict, is_pooling=False):
    def counting(prediction, is_alive):
        global count_positive, count_negative, count_tp, count_fp, count_tn, count_fn
        global log_dict

        if is_alive:
            y_test.append(0)
        else:
            y_test.append(1)

        if prediction[0] > prediction[1]:
            predictions.append(0)
            count_positive += 1

            if is_alive:
                log_dict[fpPath][image] = path
                count_fp += 1
            else:
                log_dict[tpPath][image] = path
                count_tp += 1
        else:
            predictions.append(1)
            count_negative += 1

            if is_alive:
                log_dict[tnPath][image] = path
                count_tn += 1
            else:
                log_dict[fnPath][image] = path
                count_fn += 1

    y_test = list()
    y_prob = list()
    predictions = list()

    checkpoint, load_step = create_graph(model_path, chk_point_path, FLAGS.load_step)

    with tf.Session() as sess:

        # using check point for loading graph where specific epoch
        if checkpoint:
            saver = tf.train.import_meta_graph(load_step + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)
            graph = tf.get_default_graph()

        # using pb file for loading graph (It is last step)
        else:
            graph = sess.graph

        if is_pooling:
            # infer a patient of alive
            for patient_number, image_list in image_dict["alive"].items():
                prob_alive = float()
                prob_death = float()
                count_alive = int()
                count_death = int()

                for image in image_list:
                    path = image_dict[image]
                    prediction = get_prediction(sess, graph, path, image)

                    prob_alive += prediction[0]
                    prob_death += prediction[1]

                    if prediction[0] > prediction[1]:
                        count_alive += 1
                    else:
                        count_death += 1

                y_prob.append(prob_death / (count_alive + count_death))
                counting([count_alive, count_death], is_alive=True)
            # infer a patient of death
            for patient_number, image_list in image_dict["death"].items():
                prob_alive = float()
                prob_death = float()
                count_alive = int()
                count_death = int()

                for image in image_list:
                    path = image_dict[image]
                    prediction = get_prediction(sess, graph, path, image)

                    prob_alive += prediction[0]
                    prob_death += prediction[1]

                    if prediction[0] > prediction[1]:
                        count_alive += 1
                    else:
                        count_death += 1

                y_prob.append(prob_death / (count_alive + count_death))
                counting([count_alive, count_death], is_alive=False)
        else:
            for image, path in image_dict.items():
                prediction = get_prediction(sess, graph, path, image)

                y_prob.append(prediction[-1])
                counting(prediction, is_alive=path.endswith(alivePath))

        show_scores(np.array(y_test), np.array(y_prob), np.array(predictions))


def get_prediction(sess, graph, path, image):
    image_data = tf.gfile.FastGFile(path + image, 'rb').read()
    softmax_tensor = graph.get_tensor_by_name('final_result:0')
    prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    return list(np.squeeze(prediction))


def set_new_paths(tensor_path):
    model_path = tensor_path + model_name
    save_path = tensor_path + FLAGS.save_name
    chk_point_path = tensor_path + 'checkpoints/'

    return model_path, save_path, chk_point_path


def load_log(log_path):
    try:
        with open(log_path, 'r') as read_file:
            return json.load(read_file)
    except FileNotFoundError:
        return None


def dump_log_dict(save_log_path):
    with open(save_log_path, 'w') as outfile:
        json.dump(log_dict, outfile, indent=4)
        print("\n=========================================================")
        print("\nsuccess make dump file! - file name is", save_log_path, "\n\n")


def run_inference_on_images(_):
    model_path, save_log_path, chk_point_path = set_new_paths(FLAGS.save_path)
    data_dict = load_log(FLAGS.log_path)

    if not data_dict:
        print("\nThere is no log file for testing!\n")
        return -1

    image_dict = get_images(FLAGS.image_dir, data_dict["test"])

    if FLAGS.pooling and data_dict["num_total_train"] != 0:
        print("\nPooling test!\n")
        image_dict.update(data_dict["test_dict"])
        inference_image(model_path, chk_point_path, image_dict, is_pooling=True)
    else:
        if not image_dict:
            tf.logging.fatal("File does not exist in %s", FLAGS.image_dir)
            return -1

        inference_image(model_path, chk_point_path, image_dict)
        dump_log_dict(save_log_path)


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
        '--save_path',
        type=str,
        default='',
        help='Path to saved tensor directory name.'
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default='inference.txt',
        help='Path to saved inference result file name.'
    )
    parser.add_argument(
        '--load_step',
        type=str,
        default='',
        help='Set a step to load trained inception v3 graph.'
    )
    parser.add_argument(
        '--pooling',
        type=int,
        default=0,
        help='Set a pooling method.'
    )

    # make result directory
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run_inference_on_images, argv=[sys.argv[0]] + unparsed)
