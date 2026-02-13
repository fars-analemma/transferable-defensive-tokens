#!/bin/bash
# Fine-tune the 5 transferred DefensiveToken embeddings on Llama-3.1-8B-Instruct
# using StruQ loss. Single GPU, gradient checkpointing.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp

set -a
source $BASE/.env 2>/dev/null || true
set +a
export WANDB_MODE=offline

python $BASE/aligndeftoken/transfer/tiny_adapt.py \
    --model_path $BASE/DefensiveToken/meta-llama/Llama-3.1-8B-Instruct-5TransferredTokens \
    --data_path $BASE/aligndeftoken/outputs/self_labels.json \
    --output_dir $BASE/aligndeftoken/outputs/tiny_adapt \
    --num_steps 200 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 0.1 \
    --max_length 2048 \
    --save_steps "25,50,100,150,200" \
    --seed 42

echo "=== Tiny-adapt training complete ==="
