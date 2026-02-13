#!/bin/bash
# Integrate pre-optimized DefensiveToken embeddings into both Llama-3 models.
# Runs DefensiveToken/setup.py which adds 5 special tokens to each model vocabulary.

set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate

cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/DefensiveToken

echo "=== Starting DefensiveToken setup ==="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python setup.py

echo "=== Setup complete ==="
ls -la Meta-Llama-3-8B-Instruct-5DefensiveTokens/ 2>/dev/null || echo "Llama-3 model not found"
ls -la Llama-3.1-8B-Instruct-5DefensiveTokens/ 2>/dev/null || echo "Llama-3.1 model not found"
echo "=== Done ==="
