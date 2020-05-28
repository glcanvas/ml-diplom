from executors import executor_simultaneous as ma, executor_sequential as mfa


def main_first_attention():
    args1 = """
    --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
    --algorithm_name VGG16+ATTENTION_MODULE+MLOSS+PRETRAIN_100_PRETRAIN_SUM_NO_SIGMOID
    --epochs 150 --pre_train 100 --gpu 0 --train_set 1800 --left_class_number 0
    --right_class_number 5 --seed 433 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
    --weight_decay 0.01 --model_identifier 433 --execute_from_model True
    """
    mfa.execute(args1.split())

    return
    args1 = """
    --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
    --algorithm_name VGG16+ATTENTION_MODULE+MLOSS+PRETRAIN_100_PRETRAIN_SUM_NO_SIGMOID
    --epochs 150 --pre_train 100 --gpu 0 --train_set 1800 --left_class_number 0
    --right_class_number 5 --seed 389 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
    --weight_decay 0.01 --model_identifier 389 --execute_from_model False
    """
    mfa.execute(args1.split())

    args1 = """
    --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
    --algorithm_name VGG16+ATTENTION_MODULE+MLOSS+PRETRAIN_100_PRETRAIN_SUM_NO_SIGMOID
    --epochs 150 --pre_train 100 --gpu 0 --train_set 1800 --left_class_number 0
    --right_class_number 5 --seed 198 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
    --weight_decay 0.01 --model_identifier 198 --execute_from_model False
    """
    mfa.execute(args1.split())


def main_alternate():
    args1 = """
        --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
        --algorithm_name VGG16+ATTENTION_MODULE_PRETRAIN_SUM_NO_SIGMOID
        --epochs 150 --pre_train 150 --gpu 0 --train_set 1800 --left_class_number 0
        --right_class_number 5 --seed 433 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
        --weight_decay 0.01 --model_identifier 433 --execute_from_model True
        """
    ma.execute(args1.split())

    args1 = """
        --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
        --algorithm_name VGG16+ATTENTION_MODULE_PRETRAIN_SUM_NO_SIGMOID
        --epochs 150 --pre_train 150 --gpu 0 --train_set 1800 --left_class_number 0
        --right_class_number 5 --seed 389 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
        --weight_decay 0.01 --model_identifier 389 --execute_from_model True
        """
    ma.execute(args1.split())

    args1 = """
        --run_name RUN_500_LEFT-0_RIGHT-5_TRAIN_SIZE-1800_LOOP_COUNT-3_CLR-0.001_AMLR-0.001
        --algorithm_name VGG16+ATTENTION_MODULE_PRETRAIN_SUM_NO_SIGMOID
        --epochs 150 --pre_train 150 --gpu 0 --train_set 1800 --left_class_number 0
        --right_class_number 5 --seed 198 --classifier_learning_rate 0.001 --attention_module_learning_rate 0.001 
        --weight_decay 0.01 --model_identifier 198 --execute_from_model True
        """
    ma.execute(args1.split())


if __name__ == "__main__":
    #for i in range(100):
    #    main_alternate()
    for i in range(100):
        main_first_attention()
