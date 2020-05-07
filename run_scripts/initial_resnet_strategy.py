"""
some *&@#*& with initial queue list
"""

import random

RANDOM = random.Random(0)
SEED_LIST = [RANDOM.randint(1, 500) for _ in range(3)]
ATTENTION_MODULE_LEARNING_RATES = [1e-3]
CLASS_BORDER = [(0, 5)]
RUN_NAME_RANGE_FROM = 1000
TRAIN_SIZE = 1800
EPOCHS_COUNT = 150
ALGORITHM_DATA = {
    'name': 'executor_resnet.py',
    'algorithm_name': 'RESNET_BASELINE',
    'pre_train': 200,
    'train_set': TRAIN_SIZE,
    'epochs': EPOCHS_COUNT
}
TRUST_RESNET_TYPES = ["resnet34", "resnet50", "resnet101", "resnet152"]
MEMORY_USAGE = [1600, 1600, 2400, 3000]
TRUST_LR = [1e-3, 1e-4, 1e-5]


def initial_strategy_queue_resnet(clr_idx: int = 0, resnet_type: str = None,
                                  execute_from_model: str = "false"):
    result = []
    run_id = RUN_NAME_RANGE_FROM + clr_idx
    for left_border, right_border in CLASS_BORDER:
        for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
            memory = TRUST_RESNET_TYPES.index(resnet_type)
            for seed_id in SEED_LIST:
                run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                    .format(run_id, left_border, right_border, TRAIN_SIZE, TRUST_LR[clr_idx],
                            attention_learning_rate)
                arguments = {
                    '--run_name': run_name,
                    '--algorithm_name': ALGORITHM_DATA['algorithm_name'] + "_" + resnet_type,
                    '--epochs': 150,
                    '--pre_train': 150,
                    '--train_set': 1800,
                    '--left_class_number': left_border,
                    '--right_class_number': right_border,
                    '--classifier_learning_rate': TRUST_LR[clr_idx],
                    '--attention_module_learning_rate': attention_learning_rate,
                    '--resnet_type': resnet_type,
                    '--model_identifier': seed_id,
                    '--execute_from_model': execute_from_model
                }
                result.append((ALGORITHM_DATA['name'], memory, arguments))
    return result


"""
'2;resnet101;True' '2;resnet34;False' '2;resnet50;False'

1;resnet101;True 0;resnet152;True 1;resnet152;False 
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
                commands.extend(initial_strategy_queue_resnet(int(values[0]), values[1], values[2]))
        except BaseException as e:
            print(e)
            raise e

    return commands


if __name__ == "__main__":
    r = parse_resnet_args("'1;resnet101;True' '0;resnet152;True' '1;resnet152;False'".split())
    for i in r:
        print(i)