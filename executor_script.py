import sys
import os
import executor_nvsmi as nsmi
import property as p
import time
from threading import Thread

SLEEP_SECONDS = 120
CLASSIFIER_LEARNING_RATES = [1e-3, 1e-4, 1e-5, 1e-6]
ATTENTION_MODULE_LEARNING_RATES = [1e-3, 1e-4, 1e-5, 1e-6]

PYTHON_EXECUTOR_NAME = "/home/nikita/anaconda3/envs/ml-diplom/bin/python"

CLASS_BORDER = [
    (0, 5),  # all classes
    (3, 5),  # last and prev last
    (4, 5),  # last
    (3, 4),  # last prev
]
#
# красивые презентации
#

MEMORY_USAGE = 5000  # MB
RUN_NAME_RANGE_FROM = 100
TRAIN_SIZE = 1900
EPOCHS_COUNT = 150
LOOP_COUNT = 10
PYTHON_FILE_NAME_DIR = os.path.dirname(os.path.realpath(__file__))

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


def execute_algorithm(algorithm_dict: dict, run_id: int, gpu: int, left_border: int, right_border: int):
    script_name = os.path.join(PYTHON_FILE_NAME_DIR, algorithm_dict['name'])
    run_name = "RUN_{}_LEFT:{}_RIGHT:{}_TRAIN_SIZE:{}_LOOP_COUNT:{}".format(run_id, left_border, right_border,
                                                                            TRAIN_SIZE, LOOP_COUNT)
    args = [PYTHON_EXECUTOR_NAME, script_name, '--run_name', run_name,
            '--algorithm_name', str(algorithm_dict['algorithm_name']),
            '--epochs', str(algorithm_dict['epochs']),
            '--pre_train', str(algorithm_dict['pre_train']),
            '--gpu', str(gpu),
            '--train_set', str(algorithm_dict['train_set']),
            '--left_class_number', str(left_border),
            '--right_class_number', str(right_border)
            ]
    cmd = " ".join(args)

    for i in range(LOOP_COUNT):
        os.system(cmd)


if __name__ == "__main__":

    thread_list = []
    p.initialize_log_name("NO_NUMBER", "NO_ALGORITHM", "FOR_EXEC_PURPOSE")
    run_id = RUN_NAME_RANGE_FROM
    for classifier_learning_rate in CLASSIFIER_LEARNING_RATES:
        for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
            for left_border, right_border in CLASS_BORDER:
                run_name = "RUN_{}_LEFT{}_RIGHT{}".format(run_id, left_border, right_border)
                for algorithm_data in ALGORITHM_LIST:

                    executed = False
                    while True:
                        smi = nsmi.NVLog()
                        for k in smi['Attached GPUs']:
                            gpu = int(smi['Attached GPUs'][k]['Minor Number'])
                            free_memory = int(smi['Attached GPUs'][k]['FB Memory Usage']['Free'].split()[0])
                            if free_memory < MEMORY_USAGE:
                                continue
                            executed = True
                            # HERE IN NEW THREAD
                            thread = Thread(target=execute_algorithm, args=(algorithm_data, run_id, gpu,
                                                                            left_border, right_border))
                            thread.start()
                            thread_list.append(thread)
                            break

                        if executed:
                            p.write_to_log("success start, classifier_learning_rate: {},"
                                           " attention_learning_rate: {}, left: {}, right: {}, algorithm_data: {}"
                                           " wait: {} seconds".format(classifier_learning_rate,
                                                                      attention_learning_rate, left_border,
                                                                      right_border,
                                                                      algorithm_data, SLEEP_SECONDS))
                            break
                        p.write_to_log("can't allocate memory: {}, classifier_learning_rate: {},"
                                       " attention_learning_rate: {}, left: {}, right: {}, algorithm_data: {}"
                                       " wait: {} seconds".format(
                            MEMORY_USAGE, classifier_learning_rate, attention_learning_rate, left_border, right_border,
                            algorithm_data, SLEEP_SECONDS))
                        time.sleep(SLEEP_SECONDS)
                    time.sleep(SLEEP_SECONDS)

                run_id += 1
    for t in thread_list:
        t.join()
