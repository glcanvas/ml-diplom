"""
Script for launch strategies on remove server
Which can survive out of memory errors
"""
from collections import deque
from multiprocessing import *
from datetime import datetime
import os
import random
from run_scripts import executor_nvsmi as nsmi
from utils import property as p
from utils import property_parser as pp
from utils import run_utils as ru
import time

from threading import Lock, Thread

# constants
PYTHON_EXECUTOR_NAME = "C:\\Users\\nikita\\anaconda3\\python.exe" #"/home/nduginec/nduginec_evn3/bin/python"
SLEEP_SECONDS = 120
DIPLOMA_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROPERTY_FILE = os.path.join(DIPLOMA_DIR, "executor_property.properties")

strategy_lock = Lock()
strategy_queue = deque()
thread_list = []
mapper_list = []
actual_property_index = 0
alive_process = 0


def register_commands(property_context: pp.PropertyContext):
    for i in property_context.add_list:
        script_name = i.pop('--script_name')
        script_memory = i.pop('--script_memory')
        strategy_queue.append((script_name, int(script_memory), i))


def print_status_info():
    global strategy_queue, thread_list, mapper_list

    p.write_to_log("actual_property_index={}".format(actual_property_index))
    p.write_to_log("alive_process={}".format(alive_process))
    p.write_to_log("queue:")
    for idx, i in enumerate(strategy_queue):
        p.write_to_log("idx = {}, values={}".format(idx, i))
    p.write_to_log("thread:")
    for idx, (i, j) in enumerate(zip(thread_list, mapper_list)):
        p.write_to_log("idx = {}, i={}, j={}".format(idx, i, j))
    p.write_to_log("end")


def start_strategy(executor_name: str, memory_usage: int, gpu: int, algorithms_params: dict):
    global alive_process
    executor_name = os.path.join(DIPLOMA_DIR, "executors", executor_name)
    args = [PYTHON_EXECUTOR_NAME, executor_name, "--gpu", str(gpu)]

    for k, v in algorithms_params.items():
        args.append(k)
        args.append(str(v))

    cmd = " ".join(args)

    current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    p.write_to_log("time = {} BEGIN execute: {}".format(current_time, cmd))
    status = os.system(cmd)
    current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    p.write_to_log("time = {} END execute: {}, status = {}".format(current_time, cmd, status))

    try:
        strategy_lock.acquire()
        if status != 0:
            p.write_to_log("Failed algorithm execution: {}, status={}".format(cmd, status))

            copy_alg_params = {}
            for k, v in algorithms_params.items():
                copy_alg_params[k] = v
            copy_alg_params['execute_from_model'] = 'True'
            strategy_queue.appendleft((executor_name, memory_usage, copy_alg_params))

        alive_process -= 1
    finally:
        strategy_lock.release()


def initial_strategy_queue_resnet():
    RANDOM = random.Random(0)
    SEED_LIST = [RANDOM.randint(1, 500) for _ in range(3)]
    CLASSIFIER_LEARNING_RATES = [1e-3, 1e-4, 1e-5]
    ATTENTION_MODULE_LEARNING_RATES = [1e-3]
    CLASS_BORDER = [(0, 5)]
    RUN_NAME_RANGE_FROM = 1000
    TRAIN_SIZE = 1800
    EPOCHS_COUNT = 150
    ALGORITHM_LIST = [{
        'name': 'executor_resnet.py',
        'algorithm_name': 'RESNET_BASELINE',
        'pre_train': 200,
        'memory_usage': 4000,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    }]
    RESNET_TYPES = ["resnet50", "resnet34", "resnet101", "resnet152"]
    result = []
    run_id = RUN_NAME_RANGE_FROM
    for left_border, right_border in CLASS_BORDER:
        for classifier_learning_rate in CLASSIFIER_LEARNING_RATES:
            for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
                for algorithm_data in ALGORITHM_LIST:
                    for resnet_type in RESNET_TYPES:
                        for seed_id in SEED_LIST:
                            run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                                .format(run_id, left_border, right_border, TRAIN_SIZE, classifier_learning_rate,
                                        attention_learning_rate)
                            arguments = {
                                '--run_name': run_name,
                                '--algorithm_name': algorithm_data['algorithm_name'],
                                '--epochs': 150,
                                '--pre_train': 150,
                                '--train_set': 1800,
                                '--left_class_number': left_border,
                                '--right_class_number': right_border,
                                '--classifier_learning_rate': classifier_learning_rate,
                                '--attention_module_learning_rate': attention_learning_rate,
                                '--resnet_type': resnet_type,
                                '--model_identifier': seed_id,
                                '--execute_from_model': False
                            }
                            result.append((algorithm_data['name'], algorithm_data['memory_usage'], arguments))
                    run_id += 1
    return result


def infinity_server():
    strategy_queue.extend(initial_strategy_queue_resnet())

    p.initialize_log_name("NO_NUMBER", "NO_ALGORITHM", "FOR_EXEC_PURPOSE")
    global actual_property_index, alive_process
    actual_property_context = None
    while True:
        property_context, actual_property_index = pp.process_property_file(PROPERTY_FILE, actual_property_index)
        try:
            strategy_lock.acquire()
            if property_context != actual_property_context:
                # update
                actual_property_context = property_context
                register_commands(actual_property_context)
                p.write_to_log("=" * 20)
                print_status_info()
                p.write_to_log("=" * 20)
            if len(strategy_queue) == 0:
                continue
            strategy_name, strategy_memory, strategy_arguments = strategy_queue.popleft()

            #gpu = ru.found_gpu(nsmi.NVLog(), int(strategy_memory), actual_property_context.banned_gpu,
            #                   actual_property_context.max_thread_on_gpu)
            gpu = 0
            if gpu == -1:
                strategy_queue.appendleft((strategy_name, strategy_memory, strategy_arguments))
                continue
            if alive_process >= actual_property_context.max_alive_threads:
                continue
            thread = Thread(target=start_strategy, args=(strategy_name, strategy_memory, gpu, strategy_arguments))
            thread.start()
            thread_list.append(thread)
            mapper_list.append((strategy_name, strategy_memory, gpu, strategy_arguments))
            alive_process += 1
            p.write_to_log("-" * 20)
            print_status_info()
            p.write_to_log("-" * 20)
        finally:
            strategy_lock.release()

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    infinity_server()