import os
from datetime import datetime
import sys
import argparse
import torch

# for train model
EPS = 1e-10
PROBABILITY_THRESHOLD = 0.5
TRY_CALCULATE_MODEL = 500

prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224

input_attribute = 'input'
cached_extension = '.torch'

stupid_flag = False
base_data_dir = None

if os.path.exists("/media/disk1/nduginec"):
    base_data_dir = "/media/disk1/nduginec"

elif os.path.exists("/content/gdrive/My Drive/isic"):
    base_data_dir = "/content/gdrive/My Drive/isic"

elif os.path.exists("/media/disk2/nduginec"):
    base_data_dir = "/media/disk2/nduginec"
    stupid_flag = True

elif os.path.exists("/content/drive/My Drive/isic"):
    base_data_dir = "/content/drive/My Drive/isic"

elif os.path.exists("/home/nikita/PycharmProjects"):
    base_data_dir = "/home/nikita/PycharmProjects"

else:
    raise Exception("NOT FOND BASE DIR")

data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input"
data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3"

cache_data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input/cached"
cache_data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3/cached"

log_path = base_data_dir + ("/ml-data" if stupid_flag else "") + "/logs"

log = "default_log_{}.txt".format(datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S'))


def initialize_log_name(run_number: str, algorithm_name: str, value: str, timestamp: str = None):
    global log
    if timestamp is None:
        timestamp = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    current_log_name = "log{}_{}.txt".format(timestamp, value)

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


def save_raised_model(model, epoch, identifier, run_name, algorithm_name):
    full_dir = os.path.join(base_data_dir, run_name, algorithm_name)
    os.makedirs(full_dir, exist_ok=True)
    model_filename = "id={};epoch={}".format(identifier, epoch)
    full_path = os.path.join(full_dir, model_filename)
    torch.save(model.state_dict(), full_path)


def __parse_model_name(name):
    ids = name.split(";")
    ids = {x.split("=")[0]: x.split("=")[1] for x in ids}
    return ids


def load_latest_model(identifier, run_name, algorithm_name):
    full_dir = os.path.join(base_data_dir, run_name, algorithm_name)

    models = []
    for (_, _, filenames) in os.walk(full_dir):
        models.extend(filenames)
        break

    models = list(filter(lambda x: __parse_model_name(x)['id'] == identifier, models))
    max_epoch = max(map(lambda x: int(__parse_model_name(x)['epoch']), models))
    last_model_file_name = None
    for model_file_name in models:
        full_dir_ = os.path.join(full_dir, model_file_name)
        if int(__parse_model_name(model_file_name)['epoch']) != max_epoch:
            os.remove(full_dir_)
        else:
            last_model_file_name = model_file_name
    return torch.load(os.path.join(full_dir, last_model_file_name)), max_epoch


def parse_input_commands():
    parser = argparse.ArgumentParser(description="diploma")
    parser.add_argument("--description", default="N")
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--pre_train", default=15)
    parser.add_argument("--gradient_layer_name", default="features.28")
    parser.add_argument("--from_gradient_layer", default="False")
    parser.add_argument("--epochs", default="100")
    parser.add_argument("--train_set", default="2000")
    parser.add_argument("--run_name")  # require
    parser.add_argument("--algorithm_name")  # require
    parser.add_argument("--left_class_number", default="0")  # inclusive
    parser.add_argument("--right_class_number", default="5")  # exclusive
    parser.add_argument("--classifier_learning_rate", default="1e-6")
    parser.add_argument("--attention_module_learning_rate", default="1e-4")
    parser.add_argument("--freeze_list", default="for_alternate_only")
    parser.add_argument("--seed", default="5")
    parser.add_argument("--is_freezen", default="False")
    parser.add_argument("--weight_decay", default="0")

    parser.add_argument("--time_stamp")
    parser.add_argument("--execute_from_model", default="false")
    return parser
