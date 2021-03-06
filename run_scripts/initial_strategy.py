"""
some *&@#*& with initial queue list
"""

import sys

sys.path.insert(0, "/home/nduginec/ml3/ml-diplom")
sys.path.insert(0, "/home/ubuntu/ml3/ml-diplom")

import run_scripts.initial_parser as common

ALGORITHM_DATA = [
    {
        'name': 'executor_baseline.py',
        'algorithm_name': '+baseline',
        'pre_train': common.EPOCHS_COUNT
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': '+1-1',
        'pre_train': 20
    },
    {
        'name': 'executor_sequential.py',
        'algorithm_name': '+100-50',
        'pre_train': 100
    },
    {
        'name': 'executor_simultaneous.py',
        'algorithm_name': '+am',
        'pre_train': common.EPOCHS_COUNT
    },
    {
        'name': 'executor_single_simultaneous.py',
        'algorithm_name': '*1-1',
        'pre_train': common.EPOCHS_COUNT
    },
    {
        'name': 'executor_multiple_am_single_simultaneous.py',
        'algorithm_name': '+cbam',
        'pre_train': common.EPOCHS_COUNT
    }
]

SUPPORTED_C_LR = [1e-3, 1e-4, 1e-5]
SUPPORTED_AM_LR = [1e-3, 1e-4, 1e-5]

MODEL_TYPES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg", "vgg16"]
MEMORY_USAGE = [2000, 2600, 2000, 3000, 3000, 3000, 3000]  # 4000, 6000, 6000

MODEL_STRATEGY = [x + y['algorithm_name'] for x in MODEL_TYPES for y in ALGORITHM_DATA]
MODEL_STRATEGY_DATA = [(x, y) for x in MODEL_TYPES for y in ALGORITHM_DATA]

RUN_NAME_RANGE_FROM = 1020


def initial_strategy_queue(clr_idx: int = 0,
                           amlr_idx: int = 0,
                           model_strategy: str = None,
                           execute_from_model: str = "false",
                           classifier_loss_function: str = "bceloss",
                           am_loss_function: str = "bceloss",
                           train_batch_size: str = "5",
                           test_batch_size: str = "5",
                           am_type: str = "sum",
                           dataset_type: str = None,
                           cbam_use_mloss: bool = True,
                           alpha: float = 0.5,
                           gamma: float = 0):
    result = []
    run_id = RUN_NAME_RANGE_FROM + clr_idx + amlr_idx * len(SUPPORTED_C_LR)

    clr = SUPPORTED_C_LR[clr_idx]
    amlr = SUPPORTED_AM_LR[amlr_idx]
    model_type, algo_data = MODEL_STRATEGY_DATA[MODEL_STRATEGY.index(model_strategy)]
    memory_usage = MEMORY_USAGE[MODEL_TYPES.index(model_type)]

    for left_border, right_border in common.CLASS_BORDER:
        for seed_id in common.SEED_LIST:
            run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                .format(run_id, left_border, right_border, common.TRAIN_SIZE, clr, amlr)

            algorithm_name = model_strategy + "_" + am_type + "_" + classifier_loss_function + "_" + am_loss_function
            algorithm_name += "_" + dataset_type

            if am_type == "cbam":
                algorithm_name = algorithm_name + "_" + str(cbam_use_mloss)
            if classifier_loss_function == "focal" or am_loss_function == "focal":
                algorithm_name = algorithm_name + "_" + "alpha_{}_gamma_{}".format(alpha, gamma)
            arguments = {
                '--run_name': run_name,
                '--algorithm_name': algorithm_name,
                '--epochs': common.EPOCHS_COUNT,
                '--pre_train': algo_data['pre_train'],
                '--train_set': common.TRAIN_SIZE,
                '--left_class_number': left_border,
                '--right_class_number': right_border,
                '--classifier_learning_rate': clr,
                '--attention_module_learning_rate': amlr,
                '--model_identifier': seed_id,
                '--model_type': model_type,
                '--execute_from_model': execute_from_model,
                '--classifier_loss_function': classifier_loss_function,
                '--am_loss_function': am_loss_function,
                '--am_model': am_type,
                '--train_batch_size': train_batch_size,
                '--test_batch_size': test_batch_size,
                '--dataset_type': dataset_type,
                '--cbam_use_mloss': cbam_use_mloss,
                '--alpha': alpha,
                '--gamma': gamma
            }
            result.append((algo_data['name'], memory_usage, arguments))
    return result


def parse_args(args):
    if args[0].isdecimal():
        global RUN_NAME_RANGE_FROM
        RUN_NAME_RANGE_FROM = int(args[0])
    args = args[1:]
    args_dict = common.parse_incoming_args(args)
    commands = []
    for dct in args_dict:
        if 'clr' in dct:
            clr_idx = int(dct['clr'])
            SUPPORTED_C_LR[clr_idx]
        if 'amlr' in dct:
            am_idx = int(dct['amlr'])
            SUPPORTED_AM_LR[am_idx]

        if 'model' in dct and dct['model'] in MODEL_STRATEGY:
            model_type = dct['model']

        execute_from_model = "false"
        if 'execute' in dct and (dct['execute'] == 'true' or dct['execute'] == 'false'):
            execute_from_model = dct['execute']

        if "clloss" in dct and (dct['clloss'] == 'bceloss' or dct['clloss'] == 'softf1' or dct['clloss'] == 'focal'):
            classifier_loss = dct['clloss']

        if "amloss" in dct and (dct['amloss'] == 'bceloss' or dct['amloss'] == 'softf1' or dct['amloss'] == 'focal'):
            am_loss = dct['amloss']

        if "train" in dct:
            train_batch_size = int(dct['train'])

        if "test" in dct:
            test_batch_size = int(dct['test'])

        am_type = 'product'
        if "am" in dct and (dct['am'] == "sum" or dct['am'] == 'product' or dct['am'] == 'product_shift' or dct[
            'am'] == 'sum_shift' or dct['am'] == "cbam" or dct['am'] == "conv_product" or dct['am'] == "conv_sum"):
            am_type = dct['am']

        dataset = "balanced"
        if "dataset" in dct:
            dataset = dct["dataset"]

        cbam_use_mloss = False
        if "cbam_mloss" in dct:
            cbam_use_mloss = dct["cbam_mloss"]
        alpha = 0.5
        gamma = 0
        if "alpha" in dct:
            alpha = dct['alpha']
        if 'gamma' in dct:
            gamma = dct['gamma']
        commands.extend(initial_strategy_queue(clr_idx,
                                               am_idx,
                                               model_type,
                                               execute_from_model,
                                               classifier_loss,
                                               am_loss,
                                               train_batch_size,
                                               test_batch_size,
                                               am_type,
                                               dataset,
                                               cbam_use_mloss,
                                               alpha,
                                               gamma))

    return commands


if __name__ == "__main__":
    r = parse_args(
        [
            '1488',
            'clr=1;amlr=0;dataset=balanced;model=resnet18+cbam;clloss=focal;amloss=bceloss;train=30;test=30;am=product;alpha=0.5;gamma=0'
        ])
    for i in r:
        print(i)
"""
! ! ! ! ! ! ! ! ! ! ! ! ! ! ! <- define run index
clr=0;amlr=0;dataset=balanced;model=vgg16+AM;clloss=bceloss;amloss=softf1;train=30;test=20;execute=false;am=sum
clr=0;amlr=0;dataset=balanced;model=resnet18+AM;clloss=bceloss;amloss=softf1;train=30;test=20;execute=false;

"""
