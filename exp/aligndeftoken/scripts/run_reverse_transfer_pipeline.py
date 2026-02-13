"""Reverse transfer pipeline: Llama-3-8B-Instruct (source) -> Llama-3.1-8B-Instruct (target).

Computes Procrustes alignment W* from Llama-3 to Llama-3.1, transfers DefensiveTokens,
integrates into target model, and verifies correctness. All CPU-only, no GPU required.
"""

import json
import logging
import os
import sys
import time

import numpy as np
import psutil
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from aligndeftoken.alignment.procrustes import ProcrustesAligner
from aligndeftoken.transfer.token_transfer import TransferDefensiveTokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DT_JSON = os.path.join(BASE, "DefensiveToken", "defensivetokens.json")
OUTPUT_DIR = os.path.join(BASE, "aligndeftoken", "outputs")
INTEGRATED_MODEL_DIR = os.path.join(BASE, "DefensiveToken", "meta-llama", "Llama-3.1-8B-Instruct-5TransferredTokens")

W_PATH = os.path.join(OUTPUT_DIR, "procrustes_W_llama3_to_llama31.pt")
TRANSFERRED_PATH = os.path.join(OUTPUT_DIR, "transferred_tokens_llama3_to_llama31.pt")
STATS_PATH = os.path.join(OUTPUT_DIR, "reverse_transfer_pipeline_stats.json")


def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main():
    stats = {"compute_cost": {}}
    pipeline_t0 = time.time()
    mem_peak = 0

    logger.info(f"Source: {SOURCE_MODEL}")
    logger.info(f"Target: {TARGET_MODEL}")

    t0 = time.time()
    logger.info("Loading source model embeddings (bf16, embedding-only)...")
    source_model = transformers.AutoModelForCausalLM.from_pretrained(SOURCE_MODEL, torch_dtype=torch.bfloat16)
    source_emb = source_model.get_input_embeddings().weight.data.float().cpu().numpy()
    del source_model
    import gc; gc.collect()
    logger.info(f"Source embeddings: {source_emb.shape}")

    logger.info("Loading target model embeddings (bf16, embedding-only)...")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.bfloat16)
    target_emb = target_model.get_input_embeddings().weight.data.float().cpu().numpy()
    del target_model
    gc.collect()
    logger.info(f"Target embeddings: {target_emb.shape}")

    model_load_time = time.time() - t0
    stats["compute_cost"]["model_loading_s"] = model_load_time
    stats["compute_cost"]["memory_after_model_load_MB"] = get_mem_mb()
    logger.info(f"Model loading took {model_load_time:.1f}s, mem={get_mem_mb():.0f}MB")

    aligner = ProcrustesAligner(top_k_tokens=None)
    t0 = time.time()
    alignment_stats = aligner.fit(source_emb, target_emb)
    alignment_time = time.time() - t0
    mem_peak = max(mem_peak, get_mem_mb())
    stats["alignment"] = alignment_stats
    stats["compute_cost"]["alignment_svd_s"] = alignment_time
    stats["compute_cost"]["memory_peak_alignment_MB"] = mem_peak
    logger.info(f"Alignment SVD took {alignment_time:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aligner.save(W_PATH)
    logger.info(f"Saved W* to {W_PATH}")

    del source_emb, target_emb

    transferer = TransferDefensiveTokens(
        DT_JSON,
        source_model_name=SOURCE_MODEL,
        target_model_name=TARGET_MODEL,
    )
    transferer.load_source_tokens()
    t0 = time.time()
    T_t = transferer.transfer(aligner.get_alignment_matrix())
    transfer_time = time.time() - t0
    stats["compute_cost"]["token_transfer_s"] = transfer_time
    logger.info(f"Token transfer took {transfer_time:.4f}s")

    verification = transferer.verify_transfer()
    stats["transfer_verification"] = verification

    transferer.save_transferred_embeddings(TRANSFERRED_PATH)
    logger.info(f"Saved transferred embeddings to {TRANSFERRED_PATH}")

    t0 = time.time()
    transferer.integrate_into_model(INTEGRATED_MODEL_DIR)
    integration_time = time.time() - t0
    stats["compute_cost"]["model_integration_s"] = integration_time
    logger.info(f"Model integration took {integration_time:.1f}s")

    model_verify = transferer.verify_saved_model(INTEGRATED_MODEL_DIR)
    stats["model_verification"] = model_verify

    pipeline_total = time.time() - pipeline_t0
    mem_peak = max(mem_peak, get_mem_mb())
    stats["compute_cost"]["pipeline_total_s"] = pipeline_total
    stats["compute_cost"]["memory_peak_total_MB"] = mem_peak

    transfer_method_s = alignment_time + transfer_time
    stats["compute_cost"]["comparison"] = {
        "transfer_method_s": transfer_method_s,
        "defensivetokens_optimization_gpu_hours": 16,
        "defensivetokens_optimization_description": "4 GPU-hours on 4xA100-80GB (=16 GPU-hours total)",
        "speedup_factor": f"{int(16 * 3600 / transfer_method_s)}x",
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved pipeline stats to {STATS_PATH}")

    logger.info("=== Pipeline Summary ===")
    logger.info(f"  Alignment time: {alignment_time:.1f}s")
    logger.info(f"  Transfer time: {transfer_time:.4f}s")
    logger.info(f"  Integration time: {integration_time:.1f}s")
    logger.info(f"  Total pipeline: {pipeline_total:.1f}s")
    logger.info(f"  Norm preservation OK: {verification['norm_preservation_ok']}")
    logger.info(f"  Model verification tokens_ok: {model_verify['tokens_ok']}")
    logger.info(f"  Model verification embeddings_ok: {model_verify['embeddings_ok']}")

    if not model_verify["tokens_ok"] or not model_verify["embeddings_ok"]:
        logger.error("Model verification FAILED!")
        sys.exit(1)

    logger.info("=== Reverse transfer pipeline completed successfully ===")


if __name__ == "__main__":
    main()
