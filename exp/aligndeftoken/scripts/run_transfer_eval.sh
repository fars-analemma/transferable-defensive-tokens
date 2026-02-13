#!/bin/bash
# Evaluate transferred DefensiveTokens (Llama-3.1 -> Llama-3) on AlpacaFarm prompts.
# Usage: bash run_transfer_eval.sh [--num_samples N]
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results
MODEL=$BASE/DefensiveToken/meta-llama/Meta-Llama-3-8B-Instruct-5TransferredTokens
EXTRA_ARGS="${@}"

echo "=== [Llama-3 + Transferred DT from 3.1] Attack + Benign ==="
echo "Model: $MODEL"
echo "Extra args: $EXTRA_ARGS"

python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path $RESULTS/llama3_transfer_from_31_all.json \
    $EXTRA_ARGS

echo "=== Transfer evaluation done ==="
