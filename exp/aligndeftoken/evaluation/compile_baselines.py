"""Compile all baseline results into summary JSON files and CSV table.

Reads inference outputs and computes ASR (for DefensiveTokens) and RefusalRate 
(for all methods). ASR and GCG-ASR for No Defense, Reminder, Sandwich are cited 
from DefensiveTokens Table 4.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.evaluation.asr_eval import compute_max_asr
from aligndeftoken.evaluation.refusal_eval import compute_refusal_rate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CITED_ASR = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "none": {"asr": 51.4, "gcg_asr": 94.7},
        "reminder": {"asr": 34.6, "gcg_asr": 96.6},
        "sandwich": {"asr": 56.7, "gcg_asr": 100.0},
        "defensivetokens": {"asr": 0.5, "gcg_asr": 37.5},
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "none": {"asr": 69.2, "gcg_asr": 96.2},
        "reminder": {"asr": 29.8, "gcg_asr": 97.1},
        "sandwich": {"asr": 60.6, "gcg_asr": 100.0},
        "defensivetokens": {"asr": 0.5, "gcg_asr": 24.6},
    },
}

MODEL_SHORT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}


def load_responses(path):
    with open(path) as f:
        return json.load(f)


def evaluate_refusal(data):
    return compute_refusal_rate(data["benign"]["responses"])


def evaluate_asr(data):
    attack_keys = [k for k in data if k in ("ignore", "completion", "ignore_completion")]
    if not attack_keys:
        return None
    attack_responses = {k: data[k]["responses"] for k in attack_keys}
    return compute_max_asr(attack_responses)


def main():
    print("=" * 60)
    print("Compiling Baseline Results")
    print("=" * 60)

    dt_results = {}
    prompting_results = {}

    files_config = {
        "llama3_dt_all.json": ("meta-llama/Meta-Llama-3-8B-Instruct", "defensivetokens"),
        "llama31_dt_all.json": ("meta-llama/Llama-3.1-8B-Instruct", "defensivetokens"),
        "llama3_none_benign.json": ("meta-llama/Meta-Llama-3-8B-Instruct", "none"),
        "llama3_reminder_benign.json": ("meta-llama/Meta-Llama-3-8B-Instruct", "reminder"),
        "llama3_sandwich_benign.json": ("meta-llama/Meta-Llama-3-8B-Instruct", "sandwich"),
        "llama31_none_benign.json": ("meta-llama/Llama-3.1-8B-Instruct", "none"),
        "llama31_reminder_benign.json": ("meta-llama/Llama-3.1-8B-Instruct", "reminder"),
        "llama31_sandwich_benign.json": ("meta-llama/Llama-3.1-8B-Instruct", "sandwich"),
    }

    all_metrics = []

    for fname, (model, defense) in files_config.items():
        path = os.path.join(RESULTS_DIR, fname)
        print(f"\nProcessing {fname}...")
        data = load_responses(path)

        refusal_rate = evaluate_refusal(data) if "benign" in data else None
        asr_results = evaluate_asr(data)

        cited = CITED_ASR[model][defense]
        entry = {
            "model": model,
            "model_short": MODEL_SHORT[model],
            "defense": defense,
            "cited_asr": cited["asr"],
            "cited_gcg_asr": cited["gcg_asr"],
            "refusal_rate": round(refusal_rate * 100, 1) if refusal_rate is not None else None,
        }

        if asr_results:
            entry["measured_asr"] = {k: round(v * 100, 1) for k, v in asr_results.items()}
            print(f"  Measured ASR: {entry['measured_asr']}")
            print(f"  Published ASR: {cited['asr']}%")
        else:
            entry["measured_asr"] = None

        if refusal_rate is not None:
            print(f"  RefusalRate: {refusal_rate*100:.1f}%")

        all_metrics.append(entry)

        if defense == "defensivetokens":
            dt_results[model] = entry
        else:
            prompting_results.setdefault(model, {})[defense] = entry

    dt_output = {
        "description": "DefensiveTokens baseline evaluation: ASR verification + RefusalRate measurement",
        "source": "DefensiveTokens Table 4 (ASR/GCG-ASR cited); RefusalRate newly measured",
        "results": dt_results,
    }
    dt_path = os.path.join(RESULTS_DIR, "baseline_defensivetokens.json")
    with open(dt_path, "w") as f:
        json.dump(dt_output, f, indent=2)
    print(f"\nSaved: {dt_path}")

    prompting_output = {
        "description": "Prompting baselines evaluation: RefusalRate measurement (ASR cited from paper)",
        "source": "DefensiveTokens Table 4 (ASR/GCG-ASR cited); RefusalRate newly measured",
        "results": prompting_results,
    }
    prompting_path = os.path.join(RESULTS_DIR, "baseline_prompting.json")
    with open(prompting_path, "w") as f:
        json.dump(prompting_output, f, indent=2)
    print(f"Saved: {prompting_path}")

    csv_path = os.path.join(RESULTS_DIR, "baselines_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Model", "ASR_pct", "GCG-ASR_pct", "RefusalRate_pct", "ASR_Source", "RefusalRate_Source"])
        for entry in all_metrics:
            defense_name = {
                "none": "No Defense",
                "reminder": "Reminder",
                "sandwich": "Sandwich",
                "defensivetokens": "DefensiveTokens",
            }[entry["defense"]]

            if entry["measured_asr"]:
                asr_val = entry["measured_asr"]["max_asr"]
                asr_source = f"Measured (published: {entry['cited_asr']}%)"
            else:
                asr_val = entry["cited_asr"]
                asr_source = "DefensiveTokens Table 4"

            writer.writerow([
                defense_name,
                entry["model_short"],
                asr_val,
                entry["cited_gcg_asr"],
                entry["refusal_rate"],
                asr_source,
                "Measured",
            ])
    print(f"Saved: {csv_path}")

    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Model':<25} {'ASR%':>6} {'GCG-ASR%':>9} {'RefRate%':>9} {'ASR Source'}")
    print("-" * 80)
    for entry in all_metrics:
        defense_name = {
            "none": "No Defense",
            "reminder": "Reminder",
            "sandwich": "Sandwich",
            "defensivetokens": "DefensiveTokens",
        }[entry["defense"]]

        if entry["measured_asr"]:
            asr_str = f"{entry['measured_asr']['max_asr']:>6.1f}"
            src = "Measured"
        else:
            asr_str = f"{entry['cited_asr']:>6.1f}"
            src = "Cited"

        rr_str = f"{entry['refusal_rate']:>9.1f}" if entry['refusal_rate'] is not None else "     N/A"
        print(f"{defense_name:<20} {entry['model_short']:<25} {asr_str} {entry['cited_gcg_asr']:>9.1f} {rr_str} {src}")
    print("=" * 80)


if __name__ == "__main__":
    main()
