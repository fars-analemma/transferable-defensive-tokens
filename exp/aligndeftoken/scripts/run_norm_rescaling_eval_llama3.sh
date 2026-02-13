#!/bin/bash
# Evaluate source-norm and target-norm rescaled variants for target=Llama-3
set -e

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
source .venv/bin/activate
export HF_TOKEN=YOUR_HF_TOKEN_HERE

BASEDIR=DefensiveToken/meta-llama

echo "=== Source-Norm Rescaled (3.1->3) ==="
python aligndeftoken/evaluation/inference_engine.py \
    --model_path ${BASEDIR}/Meta-Llama-3-8B-Instruct-5SourceNormTokens \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path aligndeftoken/results/llama3_sourcenorm_from_31_all.json

echo "=== Target-Norm Rescaled (3.1->3) ==="
python aligndeftoken/evaluation/inference_engine.py \
    --model_path ${BASEDIR}/Meta-Llama-3-8B-Instruct-5TargetNormTokens \
    --defense defensivetokens \
    --prompt_types benign ignore completion ignore_completion \
    --output_path aligndeftoken/results/llama3_targetnorm_from_31_all.json

echo "=== Done ==="
