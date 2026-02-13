#!/bin/bash
# Evaluate direct-copy DefensiveTokens (Llama-3.1 -> Llama-3) on AlpacaFarm.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results
MODEL=$BASE/DefensiveToken/meta-llama/Meta-Llama-3-8B-Instruct-5DirectCopyTokens

set -a
source $BASE/.env 2>/dev/null || true
set +a

echo "=== [Llama-3 + Direct Copy DT from 3.1] Attack + Benign ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path $MODEL \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path $RESULTS/llama3_directcopy_from_31_all.json

echo "=== Direct copy eval (3.1->3) done ==="
