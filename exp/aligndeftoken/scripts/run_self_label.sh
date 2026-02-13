#!/bin/bash
# Generate self-labels for Cleaned Alpaca using undefended Llama-3.1-8B-Instruct.
set -e

source /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp/.venv/bin/activate
export HF_TOKEN=YOUR_HF_TOKEN_HERE

BASE=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/transferable-defensive-tokens/exp

python3 -c "
import json, time
from vllm import LLM, SamplingParams

prompts_path = '${BASE}/aligndeftoken/outputs/self_label_prompts.json'
output_path = '${BASE}/aligndeftoken/outputs/self_labels.json'

with open(prompts_path) as f:
    data = json.load(f)

print(f'Loaded {len(data)} prompts')

llm = LLM(
    model='meta-llama/Llama-3.1-8B-Instruct',
    tensor_parallel_size=1,
    max_model_len=2048,
    trust_remote_code=True,
)

prompts = [d['prompt'] for d in data]
sampling_params = SamplingParams(temperature=0, max_tokens=512)

print(f'Running inference on {len(prompts)} prompts...')
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - t0
print(f'Inference completed in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)')

for i, output in enumerate(outputs):
    data[i]['response'] = output.outputs[0].text

with open(output_path, 'w') as f:
    json.dump(data, f)

print(f'Saved {len(data)} self-labeled samples to {output_path}')

# Quick stats
empty = sum(1 for d in data if not d['response'].strip())
print(f'Empty responses: {empty}')
avg_len = sum(len(d['response']) for d in data) / len(data)
print(f'Average response length: {avg_len:.0f} chars')
"

echo "=== Self-labeling complete ==="
