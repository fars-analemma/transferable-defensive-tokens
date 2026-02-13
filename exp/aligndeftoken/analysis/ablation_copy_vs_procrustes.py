"""Ablation: Direct Copy vs Procrustes Transfer comparison table and bar chart.

Compares No Defense, Direct Copy, Procrustes Transfer, and Full DefensiveTokens
for both transfer directions on AlpacaFarm (ASR, RefusalRate, Gap-Closed r).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(BASE, "results")


BASELINES = {
    "Llama-3.1 -> Llama-3": {
        "no_defense_asr": 51.4,
        "no_defense_refusal": 1.9,
        "full_dt_asr": 3.8,
        "full_dt_refusal": 0.5,
    },
    "Llama-3 -> Llama-3.1": {
        "no_defense_asr": 69.2,
        "no_defense_refusal": 0.5,
        "full_dt_asr": 1.9,
        "full_dt_refusal": 0.5,
    },
}

PROCRUSTES = {
    "Llama-3.1 -> Llama-3": {"asr": 0.0, "refusal": 0.5},
    "Llama-3 -> Llama-3.1": {"asr": 33.7, "refusal": 1.0},
}


def gap_closed(asr_no_defense, asr_method, asr_full_dt):
    gap = asr_no_defense - asr_full_dt
    if gap == 0:
        return 1.0
    return (asr_no_defense - asr_method) / gap


def main():
    with open(os.path.join(RESULTS, "ablation_direct_copy.json")) as f:
        dc = json.load(f)

    direct_copy = {
        "Llama-3.1 -> Llama-3": {
            "asr": dc["llama31_to_llama3"]["max_asr_pct"],
            "refusal": dc["llama31_to_llama3"]["refusal_rate_pct"],
        },
        "Llama-3 -> Llama-3.1": {
            "asr": dc["llama3_to_llama31"]["max_asr_pct"],
            "refusal": dc["llama3_to_llama31"]["refusal_rate_pct"],
        },
    }

    rows = []
    for direction in ["Llama-3.1 -> Llama-3", "Llama-3 -> Llama-3.1"]:
        b = BASELINES[direction]
        nd_asr = b["no_defense_asr"]
        fd_asr = b["full_dt_asr"]

        rows.append({
            "Direction": direction,
            "Method": "No Defense",
            "ASR (%)": nd_asr,
            "RefusalRate (%)": b["no_defense_refusal"],
            "Gap-Closed r": 0.0,
        })
        dc_asr = direct_copy[direction]["asr"]
        rows.append({
            "Direction": direction,
            "Method": "Direct Copy",
            "ASR (%)": dc_asr,
            "RefusalRate (%)": direct_copy[direction]["refusal"],
            "Gap-Closed r": round(gap_closed(nd_asr, dc_asr, fd_asr), 4),
        })
        pr_asr = PROCRUSTES[direction]["asr"]
        rows.append({
            "Direction": direction,
            "Method": "Procrustes Transfer",
            "ASR (%)": pr_asr,
            "RefusalRate (%)": PROCRUSTES[direction]["refusal"],
            "Gap-Closed r": round(gap_closed(nd_asr, pr_asr, fd_asr), 4),
        })
        rows.append({
            "Direction": direction,
            "Method": "Full DefensiveTokens",
            "ASR (%)": fd_asr,
            "RefusalRate (%)": b["full_dt_refusal"],
            "Gap-Closed r": 1.0,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS, "ablation_copy_vs_procrustes.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")
    print(df.to_string(index=False))

    improvement = {}
    for direction in ["Llama-3.1 -> Llama-3", "Llama-3 -> Llama-3.1"]:
        dc_row = df[(df["Direction"] == direction) & (df["Method"] == "Direct Copy")]
        pr_row = df[(df["Direction"] == direction) & (df["Method"] == "Procrustes Transfer")]
        dc_asr = dc_row["ASR (%)"].values[0]
        pr_asr = pr_row["ASR (%)"].values[0]
        dc_r = dc_row["Gap-Closed r"].values[0]
        pr_r = pr_row["Gap-Closed r"].values[0]
        improvement[direction] = {
            "asr_reduction": round(dc_asr - pr_asr, 1),
            "gap_closed_improvement": round(pr_r - dc_r, 4),
        }
        print(f"\n{direction}:")
        print(f"  Procrustes ASR reduction over Direct Copy: {dc_asr - pr_asr:.1f} pp")
        print(f"  Procrustes gap-closed improvement: {pr_r - dc_r:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    directions = ["Llama-3.1 -> Llama-3", "Llama-3 -> Llama-3.1"]
    methods = ["No Defense", "Direct Copy", "Procrustes Transfer", "Full DefensiveTokens"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for ax, direction in zip(axes, directions):
        sub = df[df["Direction"] == direction]
        x = np.arange(len(methods))
        asr_vals = [sub[sub["Method"] == m]["ASR (%)"].values[0] for m in methods]

        bars = ax.bar(x, asr_vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("ASR (%)", fontsize=11)
        ax.set_title(direction, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(asr_vals) * 1.15 + 2)

        for bar, val in zip(bars, asr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Ablation: Direct Copy vs Procrustes Transfer\n(AlpacaFarm, ASR %)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_path = os.path.join(RESULTS, "ablation_copy_vs_procrustes.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBar chart saved to {chart_path}")

    summary = {
        "ablation": "Direct Copy vs Procrustes Transfer",
        "benchmark": "AlpacaFarm",
        "directions": {},
    }
    for direction in directions:
        sub = df[df["Direction"] == direction]
        summary["directions"][direction] = {
            m: {
                "asr_pct": sub[sub["Method"] == m]["ASR (%)"].values[0],
                "refusal_rate_pct": sub[sub["Method"] == m]["RefusalRate (%)"].values[0],
                "gap_closed_r": sub[sub["Method"] == m]["Gap-Closed r"].values[0],
            }
            for m in methods
        }
        summary["directions"][direction]["improvement_procrustes_over_direct_copy"] = improvement[direction]

    json_path = os.path.join(RESULTS, "ablation_direct_copy.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON updated at {json_path}")


if __name__ == "__main__":
    main()
