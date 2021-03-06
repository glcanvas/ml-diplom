"""
Script for launch strategies on remove server
Which can survive out of memory errors
"""
import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

from collections import deque
from datetime import datetime
import os
from run_scripts import executor_nvsmi as nsmi, gpu_founder as ru
from utils import property as p
from utils import property_parser as pp
import time
from run_scripts import initial_strategy as inits

from threading import Lock, Thread

# constants
PYTHON_EXECUTOR_NAME = "/home/nduginec/nduginec_evn3/bin/python"  # "C:\\Users\\nikita\\anaconda3\\python.exe"
if os.path.exists("/home/nduginec/nduginec_evn3/bin/python"):
    PYTHON_EXECUTOR_NAME = "/home/nduginec/nduginec_evn3/bin/python"
elif os.path.exists("/home/nduginec/nduginetc_env3/bin/python"):
    PYTHON_EXECUTOR_NAME = "/home/nduginec/nduginetc_env3/bin/python"
elif os.path.exists("/home/ubuntu/anaconda3/bin/python"):
    PYTHON_EXECUTOR_NAME = "/home/ubuntu/anaconda3/bin/python"

    # raise Exception("Not known computer")
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

    strategy_lock.acquire()
    try:
        if status != 0:
            p.write_to_log("Failed algorithm execution: {}, status={}".format(cmd, status))

            copy_alg_params = {}
            for k, v in algorithms_params.items():
                copy_alg_params[k] = v
            copy_alg_params['--execute_from_model'] = 'True'
            strategy_queue.appendleft((executor_name, memory_usage, copy_alg_params))

        alive_process -= 1
    finally:
        strategy_lock.release()


def infinity_server(q: list):
    strategy_queue.extend(q)

    p.initialize_log_name("NO_NUMBER", "NO_ALGORITHM", "FOR_EXEC_PURPOSE")
    global actual_property_index, alive_process
    actual_property_context = None
    while True:
        property_context, actual_property_index = pp.process_property_file(PROPERTY_FILE, actual_property_index)
        strategy_lock.acquire()
        try:
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

            gpu = ru.found_gpu(nsmi.NVLog(), int(strategy_memory), actual_property_context.banned_gpu,
                               actual_property_context.max_thread_on_gpu)

            if gpu == -1:
                strategy_queue.appendleft((strategy_name, strategy_memory, strategy_arguments))
                continue
            if alive_process >= actual_property_context.max_alive_threads:
                strategy_queue.appendleft((strategy_name, strategy_memory, strategy_arguments))
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
    args = sys.argv[1:]
    q = []
    q.extend(inits.parse_args(args))
    if len(q) == 0:
        print("nothing register")

    infinity_server(q)
