#!/bin/bash
# Job 3: Run DefensiveTokens on Llama-3-8B-Instruct (attack + benign).
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results
MODEL=$BASE/DefensiveToken/meta-llama/Meta-Llama-3-8B-Instruct-5DefensiveTokens

echo "=== [Llama-3 + DT] Attack + Benign ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path $RESULTS/llama3_dt_all.json

echo "=== Llama-3 DefensiveTokens done ==="
