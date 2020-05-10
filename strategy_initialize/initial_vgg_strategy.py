"""
some *&@#*& with initial queue list
"""

import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import strategy_initialize.initial_parser as common

ALGORITHM_DATA = [
    {
        'name': 'executor_baseline.py',
        'algorithm_name': 'VGG16',
        'pre_train': common.EPOCHS_COUNT
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
        'pre_train': common.EPOCHS_COUNT
    },
    {
        'name': 'executor_single_simultaneous.py',
        'algorithm_name': 'VGG16+1-1',
        'pre_train': common.EPOCHS_COUNT
    },

]
TRUST_VGG_TYPES = ["vgg16", "vgg16+1-1", "vgg16+100-50", "vgg16+am", "vgg16+1*1"]
MEMORY_USAGE = [2000, 2000, 4000, 4000, 4000]
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
    run_id = common.RUN_NAME_RANGE_FROM + clr_idx
    for left_border, right_border in common.CLASS_BORDER:
        for attention_learning_rate in common.ATTENTION_MODULE_LEARNING_RATES:
            algo_index = TRUST_VGG_TYPES.index(vgg_type)
            for seed_id in common.SEED_LIST:
                run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                    .format(run_id, left_border, right_border, common.TRAIN_SIZE, TRUST_LR[clr_idx],
                            attention_learning_rate)
                arguments = {
                    '--run_name': run_name,
                    '--algorithm_name': ALGORITHM_DATA[algo_index]['algorithm_name'],
                    '--epochs': common.EPOCHS_COUNT,
                    '--pre_train': ALGORITHM_DATA[algo_index]['pre_train'],
                    '--train_set': common.TRAIN_SIZE,
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
    args_dict = common.parse_incoming_args(args)
    commands = []
    for dct in args_dict:
        if 'idx' in dct:
            idx = int(dct['idx'])

        if 'vgg' in dct and dct['vgg'] in TRUST_VGG_TYPES:
            vgg_type = dct['vgg']

        if 'execute' in dct and (dct['execute'] == 'true' or dct['execute'] == 'false'):
            execute_from_model = dct['execute']
        else:
            execute_from_model = "false"

        if "clloss" in dct and (dct['clloss'] == 'bceloss' or dct['clloss'] == 'softf1'):
            classifier_loss = dct['clloss']

        if "amloss" in dct and (dct['amloss'] == 'bceloss' or dct['amloss'] == 'softf1'):
            am_loss = dct['amloss']

        if "train" in dct:
            train_batch_size = int(dct['train'])

        if "test" in dct:
            test_batch_size = int(dct['test'])

        if "am" in dct and (dct['am'] == "sum" or dct['am'] == 'product'):
            am_type = dct['am']

        commands.extend(initial_strategy_queue_resnet(idx, vgg_type, execute_from_model, classifier_loss,
                                                      am_loss, train_batch_size, test_batch_size, am_type))

    return commands


"""
idx=0;execute=false;vgg=vgg16+AM;clloss=bceloss;amloss=softf1;train=30;test=20;am=sum
"""

if __name__ == "__main__":
    r = parse_vgg_args("idx=0;execute=false;vgg=vgg16+AM;clloss=bceloss;amloss=softf1;train=30;test=20;am=sum".split())
    for i in r:
        print(i)
