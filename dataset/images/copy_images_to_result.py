import json
import shutil
import os

imagePath = 'dataset/test/'                                 # 추론을 진행할 이미지 경로
resultPath = 'dataset/result/'
alivePath = 'alive/'
deathPath = 'death/'

tpPath = 'tp/'
fpPath = 'fp/'
tnPath = 'tn/'
fnPath = 'fn/'

logFullPath = 'save/inference.txt'
ctDictPath = 'log/ct_dict'


def make_result_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)


def read_log(_path):
    try:
        with open(_path, 'r') as r_file:
            return json.load(r_file)
    except FileNotFoundError:
        print("\nFile Not Found Error -", _path)


def get_patient_number(image):
    return int(image.split('_')[1])


def get_src_path(ct_dict, patient_number):
    src_path = str()

    for _path, numberList in ct_dict["test"].items():
        if patient_number in numberList:
            src_path = _path
            break

    if src_path == alivePath[:-1]:
        return alivePath
    else:
        return deathPath


def copy_images(log_dict, ct_dict):
    for dst_path, imgList in log_dict.items():
        for image in imgList:
            src_path = get_src_path(ct_dict, get_patient_number(image))
            shutil.copy(imagePath + src_path + image, resultPath + dst_path + image)


if __name__ == '__main__':
    make_result_dir(resultPath + tpPath)
    make_result_dir(resultPath + fpPath)
    make_result_dir(resultPath + tnPath)
    make_result_dir(resultPath + fnPath)

    copy_images(read_log(logFullPath), read_log(ctDictPath))
