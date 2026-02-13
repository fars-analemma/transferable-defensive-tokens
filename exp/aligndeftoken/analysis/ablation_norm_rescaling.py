"""Ablation: Norm Rescaling analysis for transferred DefensiveTokens.

Compares three conditions per direction:
  (a) Procrustes transfer (no rescaling)
  (b) Source-norm rescaling
  (c) Target-native-norm rescaling

Produces ablation_norm_rescaling.json and ablation_norm_analysis.csv.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.evaluation.asr_eval import compute_max_asr
from aligndeftoken.evaluation.refusal_eval import compute_refusal_rate

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")
OUTPUTS = os.path.join(BASE, "outputs")
DT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "DefensiveToken", "defensivetokens.json")

BASELINES = {
    "Llama-3.1 -> Llama-3": {
        "no_defense_asr": 51.4,
        "full_dt_asr": 3.8,
    },
    "Llama-3 -> Llama-3.1": {
        "no_defense_asr": 69.2,
        "full_dt_asr": 1.9,
    },
}


def eval_inference_output(path):
    with open(path) as f:
        data = json.load(f)
    results = {}
    if "benign" in data:
        results["refusal_rate"] = compute_refusal_rate(data["benign"]["responses"])
    attack_keys = [k for k in data if k in ("ignore", "completion", "ignore_completion")]
    if attack_keys:
        attack_responses = {k: data[k]["responses"] for k in attack_keys}
        asr_results = compute_max_asr(attack_responses)
        results["asr"] = asr_results
    return results


def main():
    with open(DT_PATH) as f:
        dt = json.load(f)

    directions = [
        {
            "label": "Llama-3.1 -> Llama-3",
            "source_model": "meta-llama/Llama-3.1-8B-Instruct",
            "target_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama31_to_llama3.pt"),
            "procrustes_result": os.path.join(RESULTS, "transfer_llama31_to_llama3.json"),
            "source_norm_result": os.path.join(RESULTS, "llama3_sourcenorm_from_31_all.json"),
            "target_norm_result": os.path.join(RESULTS, "llama3_targetnorm_from_31_all.json"),
        },
        {
            "label": "Llama-3 -> Llama-3.1",
            "source_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "target_model": "meta-llama/Llama-3.1-8B-Instruct",
            "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama3_to_llama31.pt"),
            "procrustes_result": os.path.join(RESULTS, "transfer_llama3_to_llama31.json"),
            "source_norm_result": os.path.join(RESULTS, "llama31_sourcenorm_from_3_all.json"),
            "target_norm_result": os.path.join(RESULTS, "llama31_targetnorm_from_3_all.json"),
        },
    ]

    all_rows = []
    json_results = {"ablation": "norm_rescaling", "directions": {}}

    for d in directions:
        label = d["label"]
        T_s = np.array(dt[d["source_model"]], dtype=np.float32)
        T_t = torch.load(d["transferred_pt"], weights_only=True).numpy().astype(np.float32)
        T_native = np.array(dt[d["target_model"]], dtype=np.float32)

        source_norms = np.linalg.norm(T_s, axis=1)
        transferred_norms = np.linalg.norm(T_t, axis=1)
        native_norms = np.linalg.norm(T_native, axis=1)

        norm_diff = np.abs(source_norms - transferred_norms)
        norm_preservation = bool(np.all(norm_diff < 1e-3))

        with open(d["procrustes_result"]) as f:
            pr = json.load(f)
        procrustes_asr = pr["max_asr_pct"]
        procrustes_refusal = pr["refusal_rate_pct"]

        src_eval = eval_inference_output(d["source_norm_result"])
        src_asr = src_eval["asr"]["max_asr"] * 100
        src_refusal = src_eval["refusal_rate"] * 100

        tgt_eval = eval_inference_output(d["target_norm_result"])
        tgt_asr = tgt_eval["asr"]["max_asr"] * 100
        tgt_refusal = tgt_eval["refusal_rate"] * 100

        source_norm_rescaled_norms = source_norms.copy()
        target_norm_rescaled_norms = native_norms.copy()

        conditions = [
            ("Procrustes (no rescaling)", transferred_norms, procrustes_asr, procrustes_refusal),
            ("Source-norm rescaling", source_norm_rescaled_norms, src_asr, src_refusal),
            ("Target-native-norm rescaling", target_norm_rescaled_norms, tgt_asr, tgt_refusal),
        ]

        dir_json = {
            "norm_preservation_verified": norm_preservation,
            "max_norm_diff_source_vs_transferred": float(np.max(norm_diff)),
            "conditions": {},
        }

        for cond_name, norms, asr, refusal in conditions:
            row = {
                "Direction": label,
                "Condition": cond_name,
                "Token L2 Norms (mean +/- std)": f"{norms.mean():.2f} +/- {norms.std():.2f}",
                "ASR (%)": round(asr, 1),
                "RefusalRate (%)": round(refusal, 1),
            }
            all_rows.append(row)
            dir_json["conditions"][cond_name] = {
                "token_norms": norms.tolist(),
                "token_norms_mean": float(norms.mean()),
                "token_norms_std": float(norms.std()),
                "asr_pct": round(asr, 1),
                "refusal_rate_pct": round(refusal, 1),
            }

        dir_json["reference_norms"] = {
            "source_dt_norms": source_norms.tolist(),
            "source_dt_mean": float(source_norms.mean()),
            "source_dt_std": float(source_norms.std()),
            "transferred_norms": transferred_norms.tolist(),
            "transferred_mean": float(transferred_norms.mean()),
            "transferred_std": float(transferred_norms.std()),
            "native_target_norms": native_norms.tolist(),
            "native_target_mean": float(native_norms.mean()),
            "native_target_std": float(native_norms.std()),
        }
        json_results["directions"][label] = dir_json

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS, "ablation_norm_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")
    print(df.to_string(index=False))

    json_path = os.path.join(RESULTS, "ablation_norm_rescaling.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    print("\n=== Key Findings ===")
    for d_label, d_data in json_results["directions"].items():
        print(f"\n{d_label}:")
        print(f"  Norm preservation (source==transferred): {d_data['norm_preservation_verified']} (max diff: {d_data['max_norm_diff_source_vs_transferred']:.2e})")
        for cond, vals in d_data["conditions"].items():
            print(f"  {cond}: ASR={vals['asr_pct']}%, Refusal={vals['refusal_rate_pct']}%, norms={vals['token_norms_mean']:.2f}+/-{vals['token_norms_std']:.2f}")


if __name__ == "__main__":
    main()
