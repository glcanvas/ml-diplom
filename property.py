import os
from datetime import datetime

prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224

input_attribute = 'input'
cached_extension = '.torch'

base_data_dir = "/home/nikita/PycharmProjects"
data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input"
data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3"

cache_data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input/cached"
cache_data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3/cached"

log_path = base_data_dir + "/logs"

current_log_name = "log{}.txt".format(datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S'))

labels_attributes = [
    'streaks',
    'negative_network',
    'milia_like_cyst',
    'globules',
    'pigment_network'
]


def write_to_log(*args):
    try:
        os.makedirs(log_path, exist_ok=True)
        log = os.path.join(log_path, current_log_name)
        with open(log, 'a+') as log_file:
            for i in args:
                log_file.write(str(i) + "\n")
            log_file.flush()
    except Exception as e:
        print("Exception while write log", e)
