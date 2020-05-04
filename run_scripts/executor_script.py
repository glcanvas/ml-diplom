from multiprocessing import Value
from datetime import datetime
import os
import random
from run_scripts import executor_nvsmi as nsmi
from utils import property as p
import time
from threading import Thread

SLEEP_SECONDS = 120
CLASSIFIER_LEARNING_RATES = [1e-3, 1e-4]
ATTENTION_MODULE_LEARNING_RATES = [1e-3]

PYTHON_EXECUTOR_NAME = "/home/nduginec/nduginec_evn3/bin/python"

CLASS_BORDER = [
    (0, 5),  # all classes
    # (3, 5),  # last and prev last
    # (4, 5),  # last
    # (3, 4),  # last prev
]

weight_decay = 0.01
RUN_NAME_RANGE_FROM = 500
TRAIN_SIZE = 1800
EPOCHS_COUNT = 150
LOOP_COUNT = 3

PYTHON_FILE_NAME_DIR = os.path.dirname(os.path.realpath(__file__))
property_file = os.path.join(PYTHON_FILE_NAME_DIR, "executor_property.properties")

r = random.Random(0)
SEED_LIST = [r.randint(1, 500) for _ in range(LOOP_COUNT)]

print(SEED_LIST)
ALGORITHM_LIST = [
    # {
    #    'name': 'executor_baseline_vgg16.py',
    #    'algorithm_name': 'VGG16',
    #    'memory_usage': 4000,
    #    'pre_train': 100,
    #    'train_set': TRAIN_SIZE,
    #    'epochs': EPOCHS_COUNT,
    # },
    {
        'name': 'executor_sequential.py',
        'algorithm_name': 'VGG16+ATTENTION_MODULE+MLOSS+PRETRAIN_100_PRETRAIN_SUM_NO_SIGMOID',
        'pre_train': 100,
        'memory_usage': 4000,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': 'VGG16+ATTENTION_MODULE+MLOSS+ALTERNATE_PRETRAIN_SUM_NO_SIGMOID',
        'pre_train': 20,
        'memory_usage': 5800,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': 'VGG16+ATTENTION_MODULE_PRETRAIN_SUM_NO_SIGMOID',
        'pre_train': EPOCHS_COUNT,
        'memory_usage': 5800,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    }
]

# [(True, Recovery Bool, i) for i in ex_list]
queue = []

MAX_ALIVE_THREADS = 8
alive_threads = Value('i', 0)


def execute_algorithm(algorithm_dict: dict, run_id: int, gpu: int, left_border: int, right_border: int,
                      classifier_learning_rate: float,
                      attention_module_learning_rate: float,
                      seed_id: int,
                      recovery: bool,
                      algorithm_position: int):
    script_name = os.path.join(PYTHON_FILE_NAME_DIR, algorithm_dict['name'])
    run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_LOOP_COUNT-{}_CLR-{}_AMLR-{}".format(run_id, left_border,
                                                                                           right_border,
                                                                                           TRAIN_SIZE, LOOP_COUNT,
                                                                                           classifier_learning_rate,
                                                                                           attention_module_learning_rate)

    args = [PYTHON_EXECUTOR_NAME, script_name,
            '--run_name', run_name,
            '--algorithm_name', str(algorithm_dict['algorithm_name']),
            '--epochs', str(algorithm_dict['epochs']),
            '--pre_train', str(algorithm_dict['pre_train']),
            '--gpu', str(gpu),
            '--train_set', str(algorithm_dict['train_set']),
            '--left_class_number', str(left_border),
            '--right_class_number', str(right_border),
            '--seed', str(seed_id),
            '--classifier_learning_rate', str(classifier_learning_rate),
            '--attention_module_learning_rate', str(attention_module_learning_rate),
            '--weight_decay', str(weight_decay),
            '--model_identifier', str(seed_id),
            '--execute_from_model', str(recovery)
            ]
    cmd = " ".join(args)
    current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    p.write_to_log("time = {} idx = {} BEGIN execute: {}".format(current_time, algorithm_position, cmd))
    status = os.system(cmd)
    current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    p.write_to_log(
        "time = {} idx = {} END execute: {}, status = {}".format(current_time, algorithm_position, cmd, status))
    alive_threads.value -= 1
    if status != 0:
        p.write_to_log("failed algorithm position: {}".format(algorithm_position))
        queue.append((True, True, build_execute_algorithm_list()[algorithm_position]))


def read_property() -> dict:
    return dict(line.strip().split('=') for line in open(property_file) if
                not line.strip().startswith('#') and len(line.strip()) > 0)


def build_execute_algorithm_list():
    algorithm_position = 0
    result = []
    run_id = RUN_NAME_RANGE_FROM
    for left_border, right_border in CLASS_BORDER:
        for classifier_learning_rate in CLASSIFIER_LEARNING_RATES:
            for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
                for algorithm_data in ALGORITHM_LIST:
                    for seed_id in SEED_LIST:
                        result.append((left_border, right_border, classifier_learning_rate, attention_learning_rate,
                                       run_id, algorithm_data, seed_id, algorithm_position))
                        algorithm_position += 1
                run_id += 1
    return result


def found_gpu(algorithm_data, smi, properties: dict) -> int:
    for k in smi['Attached GPUs']:
        gpu = int(smi['Attached GPUs'][k]['Minor Number'])
        free_memory = int(smi['Attached GPUs'][k]['FB Memory Usage']['Free'].split()[0])
        if 'banned_gpu' in properties and str(gpu) in properties['banned_gpu']:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {} is banned".format(current_time, gpu))
            continue
        if 'max_thread_on_gpu' in properties and smi['Attached GPUs'][k]['Processes'] is not None and \
                len(smi['Attached GPUs'][k]['Processes']) >= int(properties['max_thread_on_gpu']):
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {}, processes = {}, max processes={}".format(current_time, gpu,
                                                                                          len(smi['Attached GPUs'][k][
                                                                                                  'Processes']),
                                                                                          properties[
                                                                                              'max_thread_on_gpu']))
            continue

        if alive_threads.value >= MAX_ALIVE_THREADS or free_memory < algorithm_data['memory_usage']:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {}, alive_threads = {}, free memory = {}".format(current_time,
                                                                                              gpu,
                                                                                              alive_threads.value,
                                                                                              free_memory))
            continue
        return gpu
    return -1


def process_property(property_index: int, queue: list, thread_list: list, mapper_list: list, properties: dict) -> int:
    executed_list = build_execute_algorithm_list()

    for prop in properties:
        splited_prop = prop.split(".")
        if splited_prop[0] != str(property_index):
            # p.write_to_log("prop = {}, property_index = {} SKIP".format(prop, property_index))
            continue

        indexes = list(map(int, filter(lambda x: len(x) > 0, properties[prop].split(','))))

        if splited_prop[1] == "add":
            for index in indexes:
                queue.append((True, False, executed_list[index]))
            property_index += 1
        if splited_prop[1] == "remove":
            for index in indexes:
                queue[index] = (False, False, queue[index][1])

            property_index += 1
        if splited_prop[1] == "stop":
            thread_position = -1
            for index in indexes:
                for idx, i in enumerate(mapper_list):
                    if i == index:
                        thread_position = idx
                if thread_position == -1:
                    pass
                    # p.write_to_log("prop = {}, value = {} not found, skip".format(prop, index))
                else:
                    thread_list[thread_position]._stop()
            property_index += 1
    return property_index


def main_function():
    thread_list = []
    mapper_list = []
    p.initialize_log_name("NO_NUMBER", "NO_ALGORITHM", "FOR_EXEC_PURPOSE")
    p.write_to_log("seed list: ", SEED_LIST)

    ex_list = build_execute_algorithm_list()
    for idx, i in enumerate(ex_list):
        p.write_to_log("idx: {} data: {}".format(idx, i))

    execute_index = 0

    p.write_to_log("=" * 20)
    properties = read_property()
    property_index = process_property(0, queue, thread_list, mapper_list, properties)
    # for idx,(i,j,k) in enumerate(queue):
    #    queue[idx] = (i, True, k)

    p.write_to_log("execute_index={}".format(execute_index))
    p.write_to_log("property_index={}".format(property_index))
    for idx, i in enumerate(queue):
        p.write_to_log("idx={}, queue value={}".format(idx, i))
    for idx, i in enumerate(zip(thread_list, mapper_list)):
        p.write_to_log("idx={}, thread map={}".format(idx, i))
    for idx, i in enumerate(ex_list):
        p.write_to_log("idx: {} execute algorithm: {}".format(idx, i))
    p.write_to_log("=" * 20)

    while execute_index < len(queue):
        run_it, recovery, (left_border, right_border, classifier_learning_rate, attention_learning_rate, run_id,
                           algorithm_data, seed_value, algorithm_position) = queue[execute_index]
        if not run_it:
            execute_index += 1
            continue

        while True:
            time.sleep(SLEEP_SECONDS)
            #p.write_to_log("=" * 20)
            properties = read_property()
            property_index = process_property(property_index, queue, thread_list, mapper_list, properties)

            gpu = found_gpu(algorithm_data, nsmi.NVLog(), properties)
            if gpu == -1:
                continue

            alive_threads.value += 1
            thread = Thread(target=execute_algorithm, args=(algorithm_data, run_id, gpu,
                                                            left_border, right_border,
                                                            classifier_learning_rate,
                                                            attention_learning_rate,
                                                            seed_value,
                                                            recovery,
                                                            algorithm_position))
            thread.start()

            thread_list.append(thread)
            mapper_list.append(execute_index)

            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {} success start, run_id:{} classifier_learning_rate: {},"
                           " attention_learning_rate: {}, left: {}, right: {}, algorithm_data: {}"
                           " wait: {} seconds".format(current_time,
                                                      run_id, classifier_learning_rate,
                                                      attention_learning_rate, left_border,
                                                      right_border,
                                                      algorithm_data, SLEEP_SECONDS))
            p.write_to_log("=" * 20)
            break
        time.sleep(SLEEP_SECONDS)
        execute_index += 1
        p.write_to_log("execute_index={}".format(execute_index))
        p.write_to_log("property_index={}".format(property_index))
        for idx, i in enumerate(queue):
            p.write_to_log("idx={}, queue value={}".format(idx, i))
        for idx, i in enumerate(zip(thread_list, mapper_list)):
            p.write_to_log("idx={}, thread map={}".format(idx, i))
        for idx, i in enumerate(ex_list):
            p.write_to_log("idx: {} execute algorithm: {}".format(idx, i))

    for t in thread_list:
        t.join()


if __name__ == "__main__":
    main_function()
