#!/bin/bash
# Create direct-copy transferred models for both directions (CPU only).
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp
DT_JSON=$BASE/DefensiveToken/defensivetokens.json

set -a
source $BASE/.env 2>/dev/null || true
set +a

echo "=== Direction 1: Llama-3.1 -> Llama-3 (direct copy) ==="
python $BASE/aligndeftoken/transfer/direct_copy.py \
    --defensivetokens_path $DT_JSON \
    --source_model meta-llama/Llama-3.1-8B-Instruct \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir $BASE/DefensiveToken/meta-llama/Meta-Llama-3-8B-Instruct-5DirectCopyTokens

echo "=== Direction 2: Llama-3 -> Llama-3.1 (direct copy) ==="
python $BASE/aligndeftoken/transfer/direct_copy.py \
    --defensivetokens_path $DT_JSON \
    --source_model meta-llama/Meta-Llama-3-8B-Instruct \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir $BASE/DefensiveToken/meta-llama/Llama-3.1-8B-Instruct-5DirectCopyTokens

echo "=== Both direct copy models created ==="
