#!/bin/bash
# Reference launch script: 2x Tesla P40 24GB, multi-model router (llama.cpp b10453)
# Adapt: LLAMA path, config.ini path, port, slot-save path.
#
# Key points:
#  - --tensor-split 1,1 : equal split, no NVLink between P40s
#  - NCCL/P2P disabled  : P40s predate P2P over PCIe; force the safe copy path
#  - numactl interleave : spread memory across NUMA nodes
#  - --parallel 1       : ONE slot. Spec gains measured at --parallel 2 vs a
#                        --parallel 1 baseline are inflated. Keep both arms equal.
#  - --flash-attn on    : required for the MTP numbers in this repo
#  - --kv-unified       : one KV pool across models (multi-model router)
#  - --models-preset    : per-model spec flags live in config.ini sections
#  - KV cache: f16 default here (262K fits on 2x24GB with Q4/Q5 27B).
#              Add --cache-type-k q4_0 --cache-type-v q4_0 only if your context
#              does not fit in f16.
set -u
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=0
export CUDA_VISIBLE_DEVICES=0,1
export GGML_CUDA_NO_NCCL=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
ulimit -l unlimited

numactl --interleave=all /path/to/llama.cpp/build/bin/llama-server \
    -ngl all \
    --tensor-split 1,1 \
    -b 4096 \
    -ub 512 \
    --image-min-tokens 1024 \
    --fit off \
    --threads 16 \
    --threads-batch 16 \
    --host 0.0.0.0 \
    --port 10000 \
    --models-preset /path/to/config.ini \
    --cache-idle-slots \
    --slot-save-path /path/to/llama_caches \
    --ctx-checkpoints 32 \
    --slots \
    --cache-ram -1 \
    --kv-unified \
    --cont-batching \
    --parallel 1 \
    --no-warmup \
    --timeout 12000 \
    --flash-attn on
