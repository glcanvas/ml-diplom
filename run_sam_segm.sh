GPU=3
RUN_NAME=RUN_05
ALGORITHM_NAME=SAM_ATT_MODULE

SCRIPT_PATH=ml3/ml-diplom
SCRIPT_NAME=main_sam.py
EXECUTOR_NAME=~/nduginec_evn3/bin/python

DESCRIPTION=LAST_CLASS_FAIR_NO_L1

EPOCHS_COUNT=150
PRE_TRAIN=10
CHANGE_LR=5

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
