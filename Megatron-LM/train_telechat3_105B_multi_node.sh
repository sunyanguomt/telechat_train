#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export ORION ENABLE LPC=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}

export MASTER_ADDR=10.244.20.190
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7" #
export GLOO_SOCKET_IFNAME=eth0

CHECKPOINT_PATH=${1:-"./telechat3_105B_bf16"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/telechat3_105B_bf16"}
TOKENIZER_ARG=${3:-"/gfs/space/private/liuxz/code/TeleChat3-36B-Thinking"}
DATA_ARG=${4:-"/gfs/space/private/liuxz/code/bin_data/test_0"}

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=4
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=1
CP_SIZE=1
PP_SIZE=8
EP_SIZE=4
MICRO_BATCH_SIZE=1 #可修改，但不建议
GLOBAL_BATCH_SIZE=16384 #不可修改
NUM_LAYERS=45 #不可修改
DTYPE="bf16" #不可修改
SEQ_LENGTH=4096 #不可修改
MAX_POSITION_EMBEDDINGS=4096 #不可修改

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/test_cache_telechat3"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# 不可修改
MODEL_ARGS=(
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --num-layers $NUM_LAYERS \
    --hidden-size 2560 \
    --ffn-hidden-size 7680 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --q-lora-rank 1536 \
    --qk-head-dim 128 \
    --v-head-dim 128 \
    --qk-layernorm \
    --qk-pos-emb-head-dim 64 \
    --num-experts 192 \
    --moe-layer-freq [0]+[1]*44 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-dtype fp32 \
    --moe-router-score-function sigmoid \
    --moe-router-bias-update-rate 1e-4 \
    --moe-router-enable-expert-bias \
    --moe-router-topk 4 \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 2.8 \
    --moe-shared-expert-intermediate-size 1536 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-token-dispatcher-type alltoall \
    --moe-token-drop-policy probs \
    --moe-grouped-gemm \
    --seq-length  $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --use-mcore-models \
    --rope-type rope \
    --rotary-percent 1.0 \
    --rotary-base 10000 \
    --mscale 1.0 \
    --mscale-all-dim 1.0
)

# 不可修改
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
)

# Conditional arguments based on DTYPE (FP8)
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
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --expert-model-parallel-size $EP_SIZE
    --decoder-first-pipeline-num-layers 5
    --decoder-last-pipeline-num-layers 4
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --grad-reduce-in-bf16
    --manual-gc
    --empty-unused-memory-level 0
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
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
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer"
        "--trust-remote-code"
	"--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '100,0,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 131072"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 200
    --eval-iters 0
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
