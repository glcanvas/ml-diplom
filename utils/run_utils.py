"""
return index of free gpu number
"""

from utils import property as p
from datetime import datetime


def found_gpu(smi, max_algorithm_memory: int, banned_gpu: int, max_thread_on_gpu: int) -> int:
    for idx, k in enumerate(smi['Attached GPUs']):
        gpu = idx #int(smi['Attached GPUs'][k]['Minor Number'])
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
        return gpu
    return -1
