"""
some *&@#*& with initial queue list
"""

import random

RANDOM = random.Random(0)
SEED_LIST = [RANDOM.randint(1, 500) for _ in range(3)]
ATTENTION_MODULE_LEARNING_RATES = [1e-3]
CLASS_BORDER = [(0, 5)]
RUN_NAME_RANGE_FROM = 1005
TRAIN_SIZE = 1800
EPOCHS_COUNT = 150

ALGORITHM_DATA = [
    {
        'name': 'executor_baseline_vgg16.py',
        'algorithm_name': 'VGG16',
        'pre_train': EPOCHS_COUNT
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': 'VGG16+1-1',
        'pre_train': 20
    },
    {
        'name': 'executor_sequential.py',
        'algorithm_name': 'VGG16+100-50',
        'pre_train': 100
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': 'VGG16+AM',
        'pre_train': EPOCHS_COUNT
    },

]
TRUST_VGG_TYPES = ["vgg16", "vgg16+1-1", "vgg16+100-50", "vgg16+AM"]
MEMORY_USAGE = [2000, 2000, 6000, 6000]
TRUST_LR = [1e-3, 1e-4, 1e-5]


def initial_strategy_queue_resnet(clr_idx: int = 0,
                                  vgg_type: str = None,
                                  execute_from_model: str = "false",
                                  classifier_loss_function: str = "bceloss",
                                  am_loss_function: str = "bceloss",
                                  train_batch_size: str = "5",
                                  test_batch_size: str = "5",
                                  model_type: str = "sum"):
    result = []
    run_id = RUN_NAME_RANGE_FROM + clr_idx
    for left_border, right_border in CLASS_BORDER:
        for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
            algo_index = TRUST_VGG_TYPES.index(vgg_type)
            for seed_id in SEED_LIST:
                run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                    .format(run_id, left_border, right_border, TRAIN_SIZE, TRUST_LR[clr_idx],
                            attention_learning_rate)
                arguments = {
                    '--run_name': run_name,
                    '--algorithm_name': ALGORITHM_DATA[algo_index]['algorithm_name'],
                    '--epochs': 150,
                    '--pre_train': 150,
                    '--train_set': 1800,
                    '--left_class_number': left_border,
                    '--right_class_number': right_border,
                    '--classifier_learning_rate': TRUST_LR[clr_idx],
                    '--attention_module_learning_rate': attention_learning_rate,
                    '--model_identifier': seed_id,
                    '--execute_from_model': execute_from_model,
                    '--classifier_loss_function': classifier_loss_function,
                    '--am_loss_function': am_loss_function,
                    '--am_model': model_type,
                    '--train_batch_size': train_batch_size,
                    '--test_batch_size': test_batch_size,

                }
                result.append((ALGORITHM_DATA[algo_index]['name'], MEMORY_USAGE[algo_index], arguments))
    return result


def parse_vgg_args(args):
    commands = []
    for arg in args:
        arg = str(arg).lower().strip()
        try:
            values = arg.split(";")
            print(values)
            int(values[0])
            if values[1] not in TRUST_VGG_TYPES:
                continue
            if values[2] != "true" and values[2] != "false":
                continue
            if values[3] != "bceloss" and values[3] != "softf1":
                continue
            if values[4] != "bceloss" and values[4] != "softf1":
                continue
            # train
            int(values[5])
            # test
            int(values[6])
            if values[7] != "sum" and values[7] != "product":
                continue
            commands.extend(initial_strategy_queue_resnet(int(values[0]),
                                                          values[1],
                                                          values[2],
                                                          values[3],
                                                          values[4],
                                                          values[5],
                                                          values[6],
                                                          values[7]))
        except BaseException as e:
            print(e)
            raise e

    return commands


if __name__ == "__main__":
    r = parse_vgg_args("0;vgg16;False;bceloss;softf1;30;20;product".split())
    for i in r:
        print(i)
