"""Build 4 rescaled-norm model variants for the norm rescaling ablation.

Loads pre-computed Procrustes-transferred tokens + source/target-native tokens,
applies source-norm and target-native-norm rescaling, integrates into models.

Directions:
  - Llama-3.1 -> Llama-3 (source=3.1, target=3)
  - Llama-3 -> Llama-3.1 (source=3, target=3.1)
Modes: source_norm, target_native_norm
"""

import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.transfer.norm_rescaling import (
    integrate_rescaled_into_model,
    source_norm_rescale,
    target_native_norm_rescale,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
DT_PATH = os.path.join(BASE, "DefensiveToken", "defensivetokens.json")
OUTPUTS = os.path.join(BASE, "aligndeftoken", "outputs")
MODEL_BASE = os.path.join(BASE, "DefensiveToken", "meta-llama")

DIRECTIONS = [
    {
        "name": "llama31_to_llama3",
        "source_model": "meta-llama/Llama-3.1-8B-Instruct",
        "target_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama31_to_llama3.pt"),
        "source_norm_dir": os.path.join(MODEL_BASE, "Meta-Llama-3-8B-Instruct-5SourceNormTokens"),
        "target_norm_dir": os.path.join(MODEL_BASE, "Meta-Llama-3-8B-Instruct-5TargetNormTokens"),
    },
    {
        "name": "llama3_to_llama31",
        "source_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "target_model": "meta-llama/Llama-3.1-8B-Instruct",
        "transferred_pt": os.path.join(OUTPUTS, "transferred_tokens_llama3_to_llama31.pt"),
        "source_norm_dir": os.path.join(MODEL_BASE, "Llama-3.1-8B-Instruct-5SourceNormTokens"),
        "target_norm_dir": os.path.join(MODEL_BASE, "Llama-3.1-8B-Instruct-5TargetNormTokens"),
    },
]


def main():
    with open(DT_PATH) as f:
        dt = json.load(f)

    all_stats = {}

    for d in DIRECTIONS:
        logger.info(f"\n{'='*60}\nDirection: {d['name']}\n{'='*60}")

        T_t = torch.load(d["transferred_pt"], weights_only=True).numpy().astype(np.float32)
        T_s = np.array(dt[d["source_model"]], dtype=np.float32)
        T_native = np.array(dt[d["target_model"]], dtype=np.float32)

        logger.info(f"T_s shape={T_s.shape}, T_t shape={T_t.shape}, T_native shape={T_native.shape}")

        T_src_rescaled, src_stats = source_norm_rescale(T_t, T_s)
        logger.info(f"Source-norm rescaling is_noop: {src_stats['is_noop']}")

        T_tgt_rescaled, tgt_stats = target_native_norm_rescale(T_t, T_native)

        vocab_norms_info = {
            "source_norms": np.linalg.norm(T_s, axis=1).tolist(),
            "transferred_norms": np.linalg.norm(T_t, axis=1).tolist(),
            "native_target_norms": np.linalg.norm(T_native, axis=1).tolist(),
            "source_norm_rescaled_norms": np.linalg.norm(T_src_rescaled, axis=1).tolist(),
            "target_norm_rescaled_norms": np.linalg.norm(T_tgt_rescaled, axis=1).tolist(),
        }
        all_stats[d["name"]] = {
            "source_norm_stats": src_stats,
            "target_native_norm_stats": tgt_stats,
            "norm_summary": vocab_norms_info,
        }

        logger.info(f"\nIntegrating source-norm rescaled tokens into model -> {d['source_norm_dir']}")
        integrate_rescaled_into_model(T_src_rescaled, d["target_model"], d["source_norm_dir"])

        logger.info(f"\nIntegrating target-native-norm rescaled tokens into model -> {d['target_norm_dir']}")
        integrate_rescaled_into_model(T_tgt_rescaled, d["target_model"], d["target_norm_dir"])

    stats_path = os.path.join(OUTPUTS, "norm_rescaling_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nNorm rescaling stats saved to {stats_path}")

    logger.info("\n\n=== NORM SUMMARY ===")
    for d_name, s in all_stats.items():
        ns = s["norm_summary"]
        logger.info(f"\n{d_name}:")
        for label, key in [
            ("Source DT", "source_norms"),
            ("Transferred (Procrustes)", "transferred_norms"),
            ("Target Native DT", "native_target_norms"),
            ("Source-Norm Rescaled", "source_norm_rescaled_norms"),
            ("Target-Norm Rescaled", "target_norm_rescaled_norms"),
        ]:
            vals = np.array(ns[key])
            logger.info(f"  {label:30s}: mean={vals.mean():.4f} +/- {vals.std():.4f}  [{', '.join(f'{v:.4f}' for v in vals)}]")


if __name__ == "__main__":
    main()
