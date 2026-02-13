#!/bin/bash
# Evaluate transferred DefensiveTokens (Llama-3 -> Llama-3.1) on AlpacaFarm prompts.
# Runs inference with vLLM on Llama-3.1-8B-Instruct + transferred DT from Llama-3.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

export HF_TOKEN=${HF_TOKEN:?Please set HF_TOKEN environment variable}

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results
MODEL=$BASE/DefensiveToken/meta-llama/Llama-3.1-8B-Instruct-5TransferredTokens
EXTRA_ARGS="${@}"

echo "=== [Llama-3.1 + Transferred DT from 3] Attack + Benign ==="
echo "Model: $MODEL"
echo "Extra args: $EXTRA_ARGS"

python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path $RESULTS/llama31_transfer_from_3_all.json \
    $EXTRA_ARGS

echo "=== Reverse transfer evaluation done ==="
