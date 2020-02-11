GPU=1
RUN_NAME=RUN_03
ALGORITHM_NAME=SAM_LOSS

SCRIPT_PATH=ml2/ml-diplom
SCRIPT_NAME=main_sam.py
EXECUTOR_NAME=~/nduginec_evn3/bin/python

DESCRIPTION=LAST_CLASS_TWO_OPTIM_FREEZ_BASE

EPOCHS_COUNT=150
PRE_TRAIN=150
CHANGE_LR=150

# of course by default -1
CLASS_NUMBER=4

$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
$EXECUTOR_NAME ~/$SCRIPT_PATH/$SCRIPT_NAME --description $DESCRIPTION --run_name $RUN_NAME  --use_class_number $CLASS_NUMBER --algorithm_name $ALGORITHM_NAME --epochs $EPOCHS_COUNT --gpu $GPU --pre_train $PRE_TRAIN --change_lr $CHANGE_LR
