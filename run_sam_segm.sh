GPU=2
RUN_NAME=RUN_10_CL4
ALGORITHM_NAME=SAM_TWO_LOSS_WITH_AM

SCRIPT_PATH=ml1/ml-diplom
SCRIPT_NAME=main_sam.py
EXECUTOR_NAME=~/nduginec_evn3/bin/python

DESCRIPTION=LAST_CLASS_NO_FAIR

EPOCHS_COUNT=100
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
