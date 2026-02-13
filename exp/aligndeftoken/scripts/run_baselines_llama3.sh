#!/bin/bash
# Job 1: Run no-defense, reminder, sandwich baselines on Llama-3-8B-Instruct (benign only).
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

echo "=== [Llama-3] No Defense - benign ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense none \
    --prompt_types benign \
    --output_path $RESULTS/llama3_none_benign.json

echo "=== [Llama-3] Reminder - benign ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense reminder \
    --prompt_types benign \
    --output_path $RESULTS/llama3_reminder_benign.json

echo "=== [Llama-3] Sandwich - benign ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense sandwich \
    --prompt_types benign \
    --output_path $RESULTS/llama3_sandwich_benign.json

echo "=== All Llama-3 baselines done ==="
