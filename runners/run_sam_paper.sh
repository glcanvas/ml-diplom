
function prop {
    grep "${1}" ./runners.properties|cut -d'=' -f2
}

GPU=3

DESCRIPTION=CBAM_PAPER_DEFAULT_IMPL

SCRIPT_NAME=main_sam.py

ALGORITHM_NAME=BAM_CBAM

for epoch in $(seq 1 $(prop LOOP_COUNT))
do
  echo $epoch
  $(prop FULL_EXECUTOR_NAME) $(prop FULL_SCRIPT_PATH)/$SCRIPT_NAME\
    --description $DESCRIPTION\
    --run_name $(prop RUN_NAME)\
    --algorithm_name $ALGORITHM_NAME\
    --epochs $(prop EPOCHS_COUNT)\
    --gpu $GPU\
    --train_set $(prop TRAIN_SIZE)\
    --test_set $(prop TEST_SIZE)\
    --voc_items $(prop VOC_ITEMS)
done
