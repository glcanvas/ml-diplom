"""
some *&@#*& with initial queue list
"""

import random


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
                                '--algorithm_name': algorithm_data['algorithm_name'] + resnet_type,
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


def initial_strategy_queue_inception():
    RANDOM = random.Random(0)
    SEED_LIST = [RANDOM.randint(1, 500) for _ in range(3)]
    CLASSIFIER_LEARNING_RATES = [1e-3, 1e-4, 1e-5]
    ATTENTION_MODULE_LEARNING_RATES = [1e-3]
    CLASS_BORDER = [(0, 5)]
    RUN_NAME_RANGE_FROM = 1000
    TRAIN_SIZE = 1800
    EPOCHS_COUNT = 150
    ALGORITHM_LIST = [{
        'name': 'executor_inceptionv.py',
        'algorithm_name': 'INCEPTION_BASELINE',
        'pre_train': 200,
        'memory_usage': 4000,
        'train_set': TRAIN_SIZE,
        'epochs': EPOCHS_COUNT
    }]
    INCEPTION_TYPES = ["inceptionv3", "inceptionv1"]
    result = []
    run_id = RUN_NAME_RANGE_FROM
    for left_border, right_border in CLASS_BORDER:
        for classifier_learning_rate in CLASSIFIER_LEARNING_RATES:
            for attention_learning_rate in ATTENTION_MODULE_LEARNING_RATES:
                for algorithm_data in ALGORITHM_LIST:
                    for inception_type in INCEPTION_TYPES:
                        for seed_id in SEED_LIST:
                            run_name = "RUN_{}_LEFT-{}_RIGHT-{}_TRAIN_SIZE-{}_CLR-{}_AMLR-{}" \
                                .format(run_id, left_border, right_border, TRAIN_SIZE, classifier_learning_rate,
                                        attention_learning_rate)
                            image_size = 224
                            if inception_type == "inceptionv3":
                                image_size = 299

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
                                '--inception_type': inception_type,
                                '--model_identifier': seed_id,
                                '--image_size': image_size,
                                '--execute_from_model': False
                            }
                            result.append((algorithm_data['name'], algorithm_data['memory_usage'], arguments))
                    run_id += 1
    return result
