#!/bin/bash
# Reference launch script: 1x RTX 5090 32GB, multi-model router (llama.cpp b10453)
# Adapt: LLAMA path, config.ini path, port, slot-save path.
#
# Key points:
#  - --cache-type-k/v q4_0 : REQUIRED for 262K context on 32GB (Q5 weights ~19GB
#    + f16 KV for 262K does not fit). Bonus: measured ~3% faster than f16 on the
#    spec arms. If you run smaller contexts, f16 default is fine.
#  - --parallel 1          : one slot; keep A/B arms at the same --parallel.
#  - --flash-attn on       : required for the MTP numbers in this repo.
#  - --kv-unified          : one KV pool across models.
#  - --models-preset       : per-model spec flags live in config.ini sections.
#  - -b 2048 -ub 1024      : smaller batches than the 48GB box (32GB VRAM, 262K ctx).
set -u
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=0

ulimit -l unlimited

numactl --interleave=all /path/to/llama.cpp/build/bin/llama-server \
    --host 0.0.0.0 \
    --port 10000 \
    --device CUDA0 \
    --threads $(nproc) \
    --flash-attn on \
    --fit on \
    -ub 1024 \
    -b 2048 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --gpu-layers 999 \
    --parallel 1 \
    --cache-ram -1 \
    --kv-unified \
    --cont-batching \
    --slots \
    --models-preset /path/to/config.ini \
    --cache-idle-slots \
    --slot-save-path /path/to/llama_caches \
    --ctx-checkpoints 16 \
    --no-warmup \
    --timeout 12000 \
    --log-file /path/to/llama_server.log
