#!/bin/bash
# Quick sanity check: run 5 samples with no-defense on base Llama-3 model.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
RESULTS=$BASE/aligndeftoken/results

echo "=== Sanity check: no defense, 5 samples ==="
python $BASE/aligndeftoken/evaluation/inference_engine.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --defense none \
    --prompt_types benign ignore \
    --output_path $RESULTS/sanity_check.json \
    --num_samples 5 \
    --max_tokens 256

echo "=== Sanity check output ==="
python -c "
import json
with open('$RESULTS/sanity_check.json') as f:
    d = json.load(f)
print('Keys:', list(d.keys()))
for k in ['benign', 'ignore']:
    if k in d:
        print(f'{k}: {d[k][\"num_responses\"]} responses')
        for i, r in enumerate(d[k]['responses'][:3]):
            print(f'  [{i}] {r[:100]}...')
"

echo "=== Sanity check done ==="
