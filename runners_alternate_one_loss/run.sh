EXECUTOR_NAME=~/nduginec_evn3/bin/python
TRAIN_SIZE=1900

# GPU=0 RUN_NAME=RUN_30_CL_0  EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=0  TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh &
# GPU=0 RUN_NAME=RUN_31_CL_1  EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=1  TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh &
# GPU=1 RUN_NAME=RUN_32_CL_2  EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=2  TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh &
# GPU=1 RUN_NAME=RUN_33_CL_3  EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=3  TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh &
GPU=1 RUN_NAME=RUN_55_CL_4  EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=4  TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh
# GPU=2 RUN_NAME=RUN_35_CL_-1 EXECUTOR_NAME=$EXECUTOR_NAME CLASS_NUMBER=-1 TRAIN_SET_SIZE=$TRAIN_SIZE ./run_template.sh &