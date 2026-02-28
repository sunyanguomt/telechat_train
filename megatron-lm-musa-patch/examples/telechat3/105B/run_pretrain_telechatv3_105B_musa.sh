 #!/bin/bash
 pip install numpy==1.26.0
 cp /mnt/yanguo.sun/lib/libmccl* /usr/local/musa/lib
 pip install /mnt/yanguo.sun/deep_ep-1.1.0+f561e5c-cp310-cp310-linux_x86_64.whl

set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=$5
  TP_SIZE=$6
  PP_SIZE=$7
  EP_SIZE=$8
  MICRO_BATCH_SIZE=$9
  GLOBAL_BATCH_SIZE=${10}
  TOKENIZER_ARG=${11}
  RDZV_ID=${12}
set +u
# export ENABLE_PROFILER=1
# export PROFILER_FREQ=4
# export PROFILER_WARMUP_STEPS=3
# export MUSA_LAUNCH_BLOCKING=1
# export PROFILER_PROFILE_MEMORY=1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_CROSS_NIC=1
export MCCL_IB_TIMEOUT=20 
export MCCL_IB_RETRY_CNT=7

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export MUSA_BLOCK_ARBITRATION_MODE=2
export USE_MUSA_MOE=1

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
DATA_PATH=$DATA_DIR


LOG_PATH=$WORK_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$WORK_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

DTYPE="bf16"
DATA_CACHE_PATH="${PWD}/test_cache_telechat3"

export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1) # tail for cuda/ head for musa
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
echo $NODE_RANK
# export NODE_RANK=0
export MASTER_PORT=12356

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --log_dir $WORK_HOME/output_log/$RDZV_ID/$EXPNAME
    --redirects ${LOG_REDIRECTS_LEVEL:-0}
)

MODEL_ARGS=(
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --num-layers 45
    --hidden-size 2560
    --ffn-hidden-size 7680
    --num-attention-heads 32
    --kv-channels 128
    --multi-latent-attention
    --kv-lora-rank 512
    --q-lora-rank 1536
    --qk-head-dim 128
    --v-head-dim 128
    --qk-layernorm
    --qk-pos-emb-head-dim 64
    --num-experts 192
    --moe-layer-freq [0]+[1]*44
    --moe-ffn-hidden-size 1536
    --moe-router-dtype fp32
    --moe-router-score-function sigmoid
    --moe-router-bias-update-rate 1e-4
    --moe-router-enable-expert-bias
    --moe-router-topk 4
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 2.8
    --moe-shared-expert-intermediate-size 1536
    --moe-aux-loss-coeff 1e-3
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-token-drop-policy probs
    --moe-grouped-gemm
    --seq-length  4096
    --max-position-embeddings 4096
    --use-mcore-models
    --rotary-percent 1.0
    --rotary-base 10000
    --mscale 1.0
    --mscale-all-dim 1.0
    --moe-permute-fusion
    --cross-entropy-fusion-impl te
    --no-rope-fusion
    --rope-type rope
    --mtp-num-layers 1

)

# --rope-type rope
# --no-rope-fusion
#     --moe-layer-freq [0]+[1]*44

# 24414062 1T
TRAINING_ARGS=(
    --use-flash-attn
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-iters 1000
    --lr-decay-iters 1000
    --lr-warmup-iters 200 #8000
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --finetune
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
)

DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size 1
    --pipeline-model-parallel-size $PP_SIZE
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --expert-model-parallel-size $EP_SIZE
    --grad-reduce-in-bf16
    --manual-gc
    --manual-gc-interval 100
    --empty-unused-memory-level 0
    --decoder-first-pipeline-num-layers 12
    --decoder-last-pipeline-num-layers 9
)
#    --decoder-first-pipeline-num-layers 12
#    --decoder-last-pipeline-num-layers 9
# --recompute-granularity full
# --recompute-method uniform
# --recompute-num-layers 1
# --decoder-first-pipeline-num-layers 5
# --decoder-last-pipeline-num-layers 4

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
)
#     --overlap-param-gather
#    --overlap-grad-reduce
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 131072"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 4"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_PATH"
        "--tokenizer-type HuggingFaceTokenizer"
        "--trust-remote-code"
	"--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '100,0,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 4"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 131072"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 2
    --eval-iters 0
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist
    --no-save-optim
    --no-save-rng
    --distributed-timeout-minutes 60
    --tensorboard-dir "$TB_PATH"
)
# --save "$CHECKPOINT_PATH"
# --load "/mnt/yanguo.sun/dianxin/a100_ckpt/torch"

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DTYPE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS_LIST[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
    "
echo $cmd
$cmd
