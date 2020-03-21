import os
from datetime import datetime
import sys
import argparse

# for train model
EPS = 1e-10
VOC_EPS = 0.0005
PROBABILITY_THRESHOLD = 0.5
TRY_CALCULATE_MODEL = 500

prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224
voc_classes = [
    'aeroplane'  # 1
    , 'bicycle'  # 2
    , 'bird'  # 3
    , 'boat'  # 4
    , 'bottle'  # 5
    , 'bus'  # 6
    , 'car'  # 7
    , 'cat'  # 8
    , 'chair'  # 9
    , 'cow'  # 10
    , 'diningtable'  # 11
    , 'dog'  # 12
    , 'horse'  # 13
    , 'motorbike'  # 14
    , 'person'  # 15
    , 'pottedplant'  # 16
    , 'sheep'  # 17
    , 'sofa'  # 18
    , 'train'  # 19
      'tvmonitor'  # 20
]


def voc_list_to_indexes(l: list) -> list:
    list_zip = list(zip([i for i in range(1, 100)], voc_classes))
    result = []
    for item in l:
        idx, _ = list(filter(lambda x: x[1] == item, list_zip))[0]
        result.append(idx)
    return result


input_attribute = 'input'
cached_extension = '.torch'

stupid_flag = False
base_data_dir = "/home/nikita/PycharmProjects"
if os.path.exists("/media/disk1/nduginec"):
    base_data_dir = "/media/disk1/nduginec"
elif os.path.exists("/media/disk2/nduginec"):
    base_data_dir = "/media/disk2/nduginec"
    stupid_flag = True

isic_data_images_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input"
isic_data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3"

isic_cache_data_images_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input/cached"
isic_cache_data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3/cached"

voc_data_path = base_data_dir + "/VOCdevkit/VOC2012"

log_path = base_data_dir + ("/ml-data" if stupid_flag else "") + "/logs"

log = "default_log_{}.txt".format(datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S'))


def initialize_log_name(run_number: str, algorithm_name: str, value: str):
    global log
    current_log_name = "log{}_{}.txt".format(datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S'), value)

    # aaa/bbb/ccc/logs/run01
    log = os.path.join(log_path, run_number, algorithm_name)
    os.makedirs(log, exist_ok=True)
    log = os.path.join(log, current_log_name)


labels_attributes = [
    'streaks',
    'negative_network',
    'milia_like_cyst',
    'globules',
    'pigment_network'
]


def write_to_log(*args):
    try:
        with open(log, 'a+') as log_file:
            for i in args:
                log_file.write(str(i) + " ")
                print(str(i), sep=" ")
            log_file.write("\n")
            log_file.flush()
    except Exception as e:
        print("Exception while write to log", e)


def parse_input_commands():
    parser = argparse.ArgumentParser(description="diploma")
    parser.add_argument("--description", default="N")
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--pre_train", default=15)
    parser.add_argument("--gradient_layer_name", default="features.28")
    parser.add_argument("--from_gradient_layer", default="False")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--train_set", default="2000")
    parser.add_argument("--test_set", default="2000")
    parser.add_argument("--run_name")  # require
    parser.add_argument("--algorithm_name")  # require
    parser.add_argument("--voc_items", default="person")
    return parser
