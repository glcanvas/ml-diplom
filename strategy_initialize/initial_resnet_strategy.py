"""
some *&@#*& with initial queue list
"""

import random
import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import strategy_initialize.initial_parser as common

ALGORITHM_DATA = {
    'name': 'executor_resnet.py',
    'algorithm_name': 'RESNET_BASELINE',
    'pre_train': 200,
    'train_set': common.TRAIN_SIZE,
    'epochs': common.EPOCHS_COUNT
}
TRUST_RESNET_TYPES = ["resnet34", "resnet50", "resnet101", "resnet152"]
MEMORY_USAGE = [2000, 2000, 3000, 3000]
TRUST_LR = [1e-3, 1e-4, 1e-5]


def initial_strategy_queue_resnet(clr_idx: int = 0, resnet_type: str = None,
                                  execute_from_model: str = "false", loss_function: str = "bceloss"):
    result = []
    run_id = common.RUN_NAME_RANGE_FROM + clr_idx
    for left_border, right_border in common.CLASS_BORDER:
        for attention_learning_rate in common.ATTENTION_MODULE_LEARNING_RATES:
            memory_index = TRUST_RESNET_TYPES.index(resnet_type)
            for seed_id in common.SEED_LIST:
                run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                    .format(run_id, left_border, right_border, common.TRAIN_SIZE, TRUST_LR[clr_idx],
                            attention_learning_rate)
                arguments = {
                    '--run_name': run_name,
                    '--algorithm_name': ALGORITHM_DATA['algorithm_name'] + "_" + resnet_type,
                    '--epochs': common.EPOCHS_COUNT,
                    '--pre_train': ALGORITHM_DATA['pre_train'],
                    '--train_set': common.TRAIN_SIZE,
                    '--left_class_number': left_border,
                    '--right_class_number': right_border,
                    '--classifier_learning_rate': TRUST_LR[clr_idx],
                    '--attention_module_learning_rate': attention_learning_rate,
                    '--resnet_type': resnet_type,
                    '--model_identifier': seed_id,
                    '--execute_from_model': execute_from_model,
                    '--classifier_loss_function': loss_function
                }
                result.append((ALGORITHM_DATA['name'], MEMORY_USAGE[memory_index], arguments))
    return result


"""
'2;resnet101;True' '2;resnet34;False' '2;resnet50;False'

1;resnet101;False 0;resnet152;False 1;resnet152;False 
"""


def parse_resnet_args(args):
    """
    1e-3;resnet50;True
    :param args:
    :return:
    """
    commands = []
    for arg in args:
        arg = str(arg).lower().strip()
        try:
            values = arg.split(";")
            print(values)
            int(values[0])
            if values[1] not in TRUST_RESNET_TYPES:
                continue
            if values[2] == "true" or values[2] == "false":
                if values[3] == "bceloss" or values[3] == "softf1":
                    commands.extend(initial_strategy_queue_resnet(int(values[0]), values[1], values[2], values[3]))
        except BaseException as e:
            print(e)
            raise e

    return commands


if __name__ == "__main__":
    r = parse_resnet_args("'1;resnet101;True' '0;resnet152;True' '1;resnet152;False'".split())
    for i in r:
        print(i)
