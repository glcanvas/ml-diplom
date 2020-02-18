GPU=3
RUN_NAME=RUN_14_CL4
ALGORITHM_NAME=AM_FIRST_TRAIN_TWO_LOSS

SCRIPT_PATH=ml2/ml-diplom
SCRIPT_NAME=main_attention.py
EXECUTOR_NAME=~/nduginec_evn3/bin/python

DESCRIPTION=LAST_CLASS_FAIR_BIG_AM_AM_TRAIN_FIRST

EPOCHS_COUNT=150
PRE_TRAIN=100
CHANGE_LR=20

# of course by default -1
CLASS_NUMBER=4

$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
