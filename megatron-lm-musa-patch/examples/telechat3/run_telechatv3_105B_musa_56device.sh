#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=1
PP_SIZE=8
EP_SIZE=8
WORLD_SIZE=448
MICRO_BATCH_SIZE=2
# NUM_MICROBATCHES=4096
NUM_MICROBATCHES=128
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
echo $DP_SIZE
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))
echo $GLOBAL_BATCH_SIZE

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH="/mnt/yanguo.sun/dianxin/bin_data_40/test_0 /mnt/yanguo.sun/dianxin/bin_data_40/test_1 /mnt/yanguo.sun/dianxin/bin_data_40/test_2 /mnt/yanguo.sun/dianxin/bin_data_40/test_3 /mnt/yanguo.sun/dianxin/bin_data_40/test_4 /mnt/yanguo.sun/dianxin/bin_data_40/test_5 /mnt/yanguo.sun/dianxin/bin_data_40/test_6 /mnt/yanguo.sun/dianxin/bin_data_40/test_7 /mnt/yanguo.sun/dianxin/bin_data_40/test_8 /mnt/yanguo.sun/dianxin/bin_data_40/test_9 /mnt/yanguo.sun/dianxin/bin_data_40/test_10 /mnt/yanguo.sun/dianxin/bin_data_40/test_11 /mnt/yanguo.sun/dianxin/bin_data_40/test_12 /mnt/yanguo.sun/dianxin/bin_data_40/test_13 /mnt/yanguo.sun/dianxin/bin_data_40/test_14 /mnt/yanguo.sun/dianxin/bin_data_40/test_15 /mnt/yanguo.sun/dianxin/bin_data_40/test_16 /mnt/yanguo.sun/dianxin/bin_data_40/test_17 /mnt/yanguo.sun/dianxin/bin_data_40/test_18 /mnt/yanguo.sun/dianxin/bin_data_40/test_19 /mnt/yanguo.sun/dianxin/bin_data_40/test_20 /mnt/yanguo.sun/dianxin/bin_data_40/test_21 /mnt/yanguo.sun/dianxin/bin_data_40/test_22 /mnt/yanguo.sun/dianxin/bin_data_40/test_23 /mnt/yanguo.sun/dianxin/bin_data_40/test_24 /mnt/yanguo.sun/dianxin/bin_data_40/test_25 /mnt/yanguo.sun/dianxin/bin_data_40/test_26 /mnt/yanguo.sun/dianxin/bin_data_40/test_27 /mnt/yanguo.sun/dianxin/bin_data_40/test_28 /mnt/yanguo.sun/dianxin/bin_data_40/test_29 /mnt/yanguo.sun/dianxin/bin_data_40/test_30 /mnt/yanguo.sun/dianxin/bin_data_40/test_31 /mnt/yanguo.sun/dianxin/bin_data_40/test_32 /mnt/yanguo.sun/dianxin/bin_data_40/test_33 /mnt/yanguo.sun/dianxin/bin_data_40/test_34 /mnt/yanguo.sun/dianxin/bin_data_40/test_35 /mnt/yanguo.sun/dianxin/bin_data_40/test_36 /mnt/yanguo.sun/dianxin/bin_data_40/test_37 /mnt/yanguo.sun/dianxin/bin_data_40/test_38 /mnt/yanguo.sun/dianxin/bin_data_40/test_39 /mnt/yanguo.sun/dianxin/bin_data_40/test_40 "
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  TOKENIZED_MODEL=/mnt/yanguo.sun/dianxin/TeleChat3-36B-Thinking
  SCRIPT_FILE=./105B/run_pretrain_telechatv3_105B_musa.sh
  RDZV_ID=$CURRENT_TIME
set +u
echo $PATCH_HOME
cmd="bash -c 'cd $WORK_HOME; \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $EP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
    ssh $host "pkill -f '/opt/conda/envs/py310/bin/torchrun'" 
    echo "$host is killed."
done

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host 2>&1'"
  # cmd_ssh=$cmd" '"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  # echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  # ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done

