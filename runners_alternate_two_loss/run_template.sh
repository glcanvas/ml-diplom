ALGORITHM_NAME=TWO_LOSS_WITH_AM

SCRIPT_PATH=ml2/ml-diplom
SCRIPT_NAME=main_alternate.py

DESCRIPTION=TWO_LOSSES_FAIR_BIG_AM

EPOCHS_COUNT=150
PRE_TRAIN=20
CHANGE_LR=5

$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR --train_set $TRAIN_SET_SIZE
