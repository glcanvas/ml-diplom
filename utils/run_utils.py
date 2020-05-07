"""
return index of free gpu number
"""

from utils import property as p
from datetime import datetime
import os

if os.path.exists("/home/nduginec/nduginec_evn3/bin/python"):
    MAPPER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
elif os.path.exists("/home/nduginec/nduginetc_env3/bin/python"):
    MAPPER = {1: 2, 2: 1, 3: 0}
else:
    pass
    # raise Exception("Not found gpu mapper")


def found_gpu(smi, max_algorithm_memory: int, banned_gpu: int, max_thread_on_gpu: int) -> int:
    p.write_to_log("list of gpu:")
    p.write_to_log(
        [str(idx) + " " + str(smi['Attached GPUs'][gpu]['Minor Number']) + " " +
         smi['Attached GPUs'][gpu]['FB Memory Usage']['Free'].split()[0] + "| " for idx, gpu in
         enumerate(smi['Attached GPUs'])])
    p.write_to_log("Mapper=", MAPPER)
    for idx, k in enumerate(smi['Attached GPUs']):
        gpu = int(smi['Attached GPUs'][k]['Minor Number'])
        free_memory = int(smi['Attached GPUs'][k]['FB Memory Usage']['Free'].split()[0])
        if banned_gpu == gpu:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {} is banned".format(current_time, gpu))
            continue
        if smi['Attached GPUs'][k]['Processes'] is not None and len(
                smi['Attached GPUs'][k]['Processes']) >= max_thread_on_gpu:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {} has processes = {} but max processes = {}".format(current_time, gpu,
                                                                                                  len(smi[
                                                                                                          'Attached GPUs']
                                                                                                      [k]['Processes']),
                                                                                                  max_thread_on_gpu))
            continue

        if free_memory < max_algorithm_memory:
            current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
            p.write_to_log("time = {}, gpu = {} has free memory = {}, but required = {}".format(current_time,
                                                                                                gpu,
                                                                                                free_memory,
                                                                                                max_algorithm_memory))
            continue
        current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
        p.write_to_log("time = {} found gpu = {}".format(current_time, MAPPER[gpu]))
        return MAPPER[gpu]
    current_time = datetime.today().strftime('%Y-%m-%d-_-%H_%M_%S')
    p.write_to_log("time = {} not found gpu".format(current_time))
    return -1
