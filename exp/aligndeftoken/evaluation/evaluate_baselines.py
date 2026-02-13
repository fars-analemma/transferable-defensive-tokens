"""Evaluate baseline inference outputs: compute ASR and RefusalRate.

Reads JSON outputs from inference_engine.py and produces metric summaries.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.evaluation.asr_eval import compute_max_asr
from aligndeftoken.evaluation.refusal_eval import compute_refusal_rate


def evaluate_file(path):
    with open(path) as f:
        data = json.load(f)

    results = {
        "model_path": data["model_path"],
        "defense": data["defense"],
        "num_samples": data["num_samples"],
    }

    if "benign" in data:
        benign_responses = data["benign"]["responses"]
        results["refusal_rate"] = compute_refusal_rate(benign_responses)
        results["num_benign"] = len(benign_responses)

    attack_keys = [k for k in data if k in ("ignore", "completion", "ignore_completion")]
    if attack_keys:
        attack_responses = {k: data[k]["responses"] for k in attack_keys}
        asr_results = compute_max_asr(attack_responses)
        results["asr"] = asr_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline inference outputs")
    parser.add_argument("--input_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    all_results = []
    for path in args.input_files:
        print(f"\nEvaluating {path}...")
        result = evaluate_file(path)
        all_results.append(result)
        print(f"  Model: {result['model_path']}")
        print(f"  Defense: {result['defense']}")
        if "refusal_rate" in result:
            print(f"  RefusalRate: {result['refusal_rate']:.4f} ({result['refusal_rate']*100:.1f}%)")
        if "asr" in result:
            for k, v in result["asr"].items():
                print(f"  ASR ({k}): {v:.4f} ({v*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output_path}")


if __name__ == "__main__":
    main()
