#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export nnodes="${NNODES:-1}"
export nproc_per_node="${NPROC_PER_NODE:-1}"
export node_rank="${NODE_RANK:-0}"
export master_addr="${MASTER_ADDR:-127.0.0.1}"
export master_port="${MASTER_PORT:-29954}"

torchrun \
  --nnodes=$nnodes \
  --nproc_per_node=$nproc_per_node \
  --node_rank=$node_rank \
  --master_addr=$master_addr \
  --master_port=$master_port \
  tokenizer/tokenizer_image/training/train_vq.py "$@"
