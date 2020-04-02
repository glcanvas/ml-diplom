import sys
from multiprocessing import Pool, Value
import subprocess
from datetime import datetime
import os
import random
import executor_nvsmi as nsmi
import property as p
import time
from threading import Thread
import threading
import inspect
import ctypes


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


SLEEP_SECONDS = 120
CLASSIFIER_LEARNING_RATES = [1e-3, 1e-4, 1e-5, 1e-6]
ATTENTION_MODULE_LEARNING_RATES = [1e-3]

PYTHON_EXECUTOR_NAME = "/home/nduginec/nduginec_evn3/bin/python"

CLASS_BORDER = [
    (0, 5),  # all classes
    (3, 5),  # last and prev last
    (4, 5),  # last
    (3, 4),  # last prev
]

MEMORY_USAGE = 5000  # MB
RUN_NAME_RANGE_FROM = 500
TRAIN_SIZE = 1800
EPOCHS_COUNT = 150
LOOP_COUNT = 8
PYTHON_FILE_NAME_DIR = os.path.dirname(os.path.realpath(__file__))
property_file = os.path.join(PYTHON_FILE_NAME_DIR, "executor_property.properties")

r = random.Random(0)
SEED_LIST = [r.randint(1, 500) for _ in range(LOOP_COUNT)]

print(SEED_LIST)
ALGORITHM_LIST = [
    {
        'name': 'main_default_classifier.py',
        'algorithm_name': 'VGG16',
        'pre_train': 100,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT,
    },
    {
        'name': 'main_first_attention.py',
        'algorithm_name': 'AM_AT_FIRST_THEN_CL_TWO_LOSS',
        'pre_train': 100,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    },
    {
        'name': 'main_alternate.py',
        'algorithm_name': 'TWO_LOSS_WITH_AM',
        'pre_train': 20,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    },
    {
        'name': 'main_alternate.py',
        'algorithm_name': 'ONE_LOSS_WITH_AM',
        'pre_train': EPOCHS_COUNT,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    }
]

MAX_ALIVE_THREADS = 8
alive_threads = Value('i', 0)


def execute_algorithm(algorithm_dict: dict, run_id: int, gpu: int, left_border: int, right_border: int,
                      classifier_learning_rate: float,
                      attention_module_learning_rate: float):
    script_name = os.path.join(PYTHON_FILE_NAME_DIR, algorithm_dict['name'])
    run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_LOOP_COUNT-{}_CLR-{}_AMLR-{}".format(run_id, left_border,
                                                                                           right_border,
                                                                                           TRAIN_SIZE, LOOP_COUNT,
                                                                                           classifier_learning_rate,
                                                                                           attention_module_learning_rate)

    for i in range(LOOP_COUNT):
        args = [PYTHON_EXECUTOR_NAME, script_name, '--run_name', run_name,
                '--algorithm_name', str(algorithm_dict['algorithm_name']),
                '--epochs', str(algorithm_dict['epochs']),
                '--pre_train', str(algorithm_dict['pre_train']),
                '--gpu', str(gpu),
                '--train_set', str(algorithm_dict['train_set']),
                '--left_class_number', str(left_border),
                '--right_class_number', str(right_border),
                '--seed', str(SEED_LIST[i]),
                '--classifier_learning_rate', str(classifier_learning_rate),
                '--attention_module_learning_rate', str(attention_module_learning_rate)
                ]
        cmd = " ".join(args)
        current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
        p.write_to_log("time = {} idx = {} BEGIN execute: {}".format(current_time, i, cmd))
        os.system(cmd)
        current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
        p.write_to_log("time = {} idx = {} END execute: {}".format(current_time, i, cmd))
    alive_threads.value -= 1


def read_property() -> dict:
    return dict(
        line.strip().split('=') for line in open(property_file) if
        not line.strip().startswith('#') and len(line.strip()) > 0)


def build_execute_list():
    result = []
    run_id = RUN_NAME_RANGE_FROM
    for left_border, right_border in CLASS_BORDER:
        for classifier_learning_rate in CLASSIFIER_LEARNING_RATES:
            for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
                for algorithm_data in ALGORITHM_LIST:
                    result.append((left_border, right_border, classifier_learning_rate, attention_learning_rate,
                                   run_id, algorithm_data))
                run_id += 1
    return result


def found_gpu(smi, properties: dict) -> int:
    for k in smi['Attached GPUs']:
        gpu = int(smi['Attached GPUs'][k]['Minor Number'])
        free_memory = int(smi['Attached GPUs'][k]['FB Memory Usage']['Free'].split()[0])
        if 'banned_gpu' in properties and str(gpu) in properties['banned_gpu']:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {} is banned".format(current_time, gpu))
            continue
        if 'max_thread_on_gpu' in properties and smi['Attached GPUs'][k]['Processes'] is not None and \
                len(smi['Attached GPUs'][k]['Processes']) > int(properties['max_thread_on_gpu']):
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {}, processes = {}, max processes={}".format(current_time, gpu,
                                                                                          alive_threads.value,
                                                                                          properties[
                                                                                              'max_thread_on_gpu']))
            continue

        if alive_threads.value >= MAX_ALIVE_THREADS or free_memory < MEMORY_USAGE:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {}, alive_threads = {}, free memory = {}".format(current_time,
                                                                                              gpu,
                                                                                              alive_threads.value,
                                                                                              free_memory))
            continue
        return gpu
    return -1


def process_property(property_index: int, queue: list, thread_list: list, mapper_list: list, properties: dict) -> int:
    executed_list = build_execute_list()

    for prop in properties:
        splited_prop = prop.split(".")
        if splited_prop[0] != str(property_index):
            p.write_to_log("prop = {}, property_index = {} SKIP".format(prop, property_index))
            continue

        index = int(properties[prop])
        if splited_prop[1] == "add":
            queue.append((True, executed_list[index]))
            property_index += 1
        if splited_prop[1] == "remove":
            queue[index] = (False, queue[index][1])
            property_index += 1
        if splited_prop[1] == "stop":
            thread_position = -1
            for idx, i in enumerate(mapper_list):
                if i == index:
                    thread_position = idx
            if thread_position == -1:
                p.write_to_log("prop = {}, value = {} not found, skip".format(prop, index))
            else:
                thread_list[thread_position]._stop()
            property_index += 1
    return property_index


if __name__ == "__main__":

    thread_list = []
    mapper_list = []
    p.initialize_log_name("NO_NUMBER", "NO_ALGORITHM", "FOR_EXEC_PURPOSE")
    p.write_to_log("seed list: ", SEED_LIST)

    ex_list = build_execute_list()
    for idx, i in enumerate(ex_list):
        p.write_to_log("idx: {} data: {}".format(idx, i))

    queue = [] # [(True, i) for i in ex_list]
    execute_index = 0
    property_index = 0
    while execute_index < len(queue):
        run_it, (left_border, right_border, classifier_learning_rate, attention_learning_rate, run_id,
                 algorithm_data) = queue[execute_index]

        while True:
            p.write_to_log("=" * 20)
            properties = read_property()

            property_index = process_property(property_index, queue, thread_list, mapper_list, properties)
            p.write_to_log("execute_index={}".format(execute_index))
            p.write_to_log("property_index={}".format(property_index))
            for idx, i in enumerate(queue):
                p.write_to_log("idx={}, queue value={}".format(idx, i))
            for idx, i in enumerate(zip(thread_list, mapper_list)):
                p.write_to_log("idx={}, thread map={}".format(idx, i))
            for idx, i in enumerate(ex_list):
                p.write_to_log("idx: {} execute algorithm: {}".format(idx, i))
            gpu = found_gpu(nsmi.NVLog(), properties)
            if gpu == -1:
                current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
                p.write_to_log("time = {} can't allocate memory: {}, classifier_learning_rate: {},"
                               " attention_learning_rate: {}, left: {}, right: {}, algorithm_data: {}"
                               " wait: {} seconds".format(current_time, MEMORY_USAGE, classifier_learning_rate,
                                                          attention_learning_rate, left_border,
                                                          right_border, algorithm_data, SLEEP_SECONDS))
                time.sleep(SLEEP_SECONDS)
                p.write_to_log("=" * 20)
                continue

            alive_threads.value += 1
            thread = Thread(target=execute_algorithm, args=(algorithm_data, run_id, gpu,
                                                            left_border, right_border,
                                                            classifier_learning_rate,
                                                            attention_learning_rate))
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

    for t in thread_list:
        t.join()
