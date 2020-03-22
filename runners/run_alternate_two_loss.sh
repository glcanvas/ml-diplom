
function prop {
    grep "${1}" ./runners.properties|cut -d'=' -f2
}

GPU=2

DESCRIPTION=TWO_LOSSES_FAIR_BIG_AM_DEFAULT_LOSS

SCRIPT_NAME=main_alternate.py

ALGORITHM_NAME=TWO_LOSS_WITH_AM
PRE_TRAIN_EPOCHS=20

for epoch in $(seq 1 $(prop LOOP_COUNT))
do
  echo $epoch
  $(prop FULL_EXECUTOR_NAME) $(prop FULL_SCRIPT_PATH)/$SCRIPT_NAME\
    --description $DESCRIPTION\
    --run_name $(prop RUN_NAME)\
    --algorithm_name $ALGORITHM_NAME\
    --epochs $(prop EPOCHS_COUNT)\
    --pre_train $PRE_TRAIN_EPOCHS
    --gpu $GPU\
    --train_set $(prop TRAIN_SIZE)\
    --left_class_number $(prop LEFT_CLASS)\
    --right_class_number $(prop RIGHT_CLASS)
done
