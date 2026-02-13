"""Compute ASR and RefusalRate from direct-copy inference outputs and save to ablation_direct_copy.json."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.evaluation.asr_eval import compute_max_asr
from aligndeftoken.evaluation.refusal_eval import compute_refusal_rate

BASE = os.path.join(os.path.dirname(__file__), "..", "results")


def evaluate_file(path):
    with open(path) as f:
        data = json.load(f)

    attack_keys = [k for k in data if k in ("ignore", "completion", "ignore_completion")]
    attack_responses = {k: data[k]["responses"] for k in attack_keys}
    asr_results = compute_max_asr(attack_responses)

    benign_responses = data["benign"]["responses"]
    refusal_rate = compute_refusal_rate(benign_responses)

    return {
        "model_path": data["model_path"],
        "defense": data["defense"],
        "num_samples": data["num_samples"],
        "asr_per_variant": {k: round(v, 4) for k, v in asr_results.items() if k != "max_asr"},
        "max_asr": round(asr_results["max_asr"], 4),
        "max_asr_pct": round(asr_results["max_asr"] * 100, 1),
        "refusal_rate": round(refusal_rate, 4),
        "refusal_rate_pct": round(refusal_rate * 100, 1),
    }


results = {}

print("=== Direction: Llama-3.1 -> Llama-3 (direct copy) ===")
r1 = evaluate_file(os.path.join(BASE, "llama3_directcopy_from_31_all.json"))
results["llama31_to_llama3"] = r1
print(f"  ASR (max): {r1['max_asr_pct']}%")
print(f"  RefusalRate: {r1['refusal_rate_pct']}%")
for k, v in r1["asr_per_variant"].items():
    print(f"  ASR ({k}): {v*100:.1f}%")

print("\n=== Direction: Llama-3 -> Llama-3.1 (direct copy) ===")
r2 = evaluate_file(os.path.join(BASE, "llama31_directcopy_from_3_all.json"))
results["llama3_to_llama31"] = r2
print(f"  ASR (max): {r2['max_asr_pct']}%")
print(f"  RefusalRate: {r2['refusal_rate_pct']}%")
for k, v in r2["asr_per_variant"].items():
    print(f"  ASR ({k}): {v*100:.1f}%")

output_path = os.path.join(BASE, "ablation_direct_copy.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
