"""End-to-end Procrustes transfer pipeline: Llama-3.1 -> Llama-3 DefensiveTokens.

Runs entirely on CPU. Computes alignment, transfers tokens, integrates into
target model, and records compute cost (wall-clock + memory).
"""

import json
import logging
import os
import sys
import time
import tracemalloc

import numpy as np
import torch
import transformers


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aligndeftoken.alignment.procrustes import ProcrustesAligner
from aligndeftoken.transfer.token_transfer import TransferDefensiveTokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DT_JSON = os.path.join(BASE, "DefensiveToken", "defensivetokens.json")
OUTPUT_DIR = os.path.join(BASE, "aligndeftoken", "outputs")
TRANSFERRED_MODEL_DIR = os.path.join(BASE, "DefensiveToken", "meta-llama", "Meta-Llama-3-8B-Instruct-5TransferredTokens")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compute_cost = {}
    all_stats = {}

    tracemalloc.start()
    pipeline_start = time.time()

    # --- Step 1: Load models sequentially and extract embeddings ---
    import gc
    logger.info("=== Step 1: Loading models and extracting embeddings ===")
    t0 = time.time()
    aligner = ProcrustesAligner(top_k_tokens=None)

    logger.info(f"Loading source model: {SOURCE_MODEL} (embeddings only, bf16)")
    source_model = transformers.AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16,
    )
    E_s = aligner.extract_embeddings(source_model)
    del source_model
    gc.collect()
    logger.info("Freed source model memory")

    logger.info(f"Loading target model: {TARGET_MODEL} (embeddings only, bf16)")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16,
    )
    E_t = aligner.extract_embeddings(target_model)
    del target_model
    gc.collect()
    logger.info("Freed target model memory")

    compute_cost["model_loading_s"] = time.time() - t0
    mem_after_load = tracemalloc.get_traced_memory()
    compute_cost["memory_after_model_load_MB"] = mem_after_load[1] / 1024 / 1024
    logger.info(f"Model loading + extraction: {compute_cost['model_loading_s']:.1f}s, peak mem: {mem_after_load[1]/1024/1024:.0f} MB")

    # --- Step 2: Compute Procrustes alignment ---
    logger.info("=== Step 2: Computing Procrustes alignment ===")

    t0 = time.time()
    alignment_stats = aligner.fit(E_s, E_t)
    compute_cost["alignment_svd_s"] = time.time() - t0
    mem_after_align = tracemalloc.get_traced_memory()
    compute_cost["memory_peak_alignment_MB"] = mem_after_align[1] / 1024 / 1024
    all_stats["alignment"] = alignment_stats

    assert alignment_stats["orthogonality_error"] < 1e-4, \
        f"W* is not orthogonal: error={alignment_stats['orthogonality_error']}"
    logger.info("Orthogonality check PASSED")

    W_path = os.path.join(OUTPUT_DIR, "procrustes_W_llama31_to_llama3.pt")
    aligner.save(W_path)

    # --- Step 3: Transfer DefensiveTokens ---
    logger.info("=== Step 3: Transferring DefensiveTokens ===")
    transferer = TransferDefensiveTokens(
        defensivetokens_path=DT_JSON,
        source_model_name=SOURCE_MODEL,
        target_model_name=TARGET_MODEL,
    )
    transferer.load_source_tokens()

    t0 = time.time()
    T_t = transferer.transfer(aligner.get_alignment_matrix())
    compute_cost["token_transfer_s"] = time.time() - t0
    logger.info(f"Token transfer: {compute_cost['token_transfer_s']:.4f}s")

    transfer_verify = transferer.verify_transfer()
    all_stats["transfer_verification"] = transfer_verify
    assert transfer_verify["norm_preservation_ok"], "Norm preservation failed after transfer"
    logger.info("Norm preservation check PASSED")

    emb_path = os.path.join(OUTPUT_DIR, "transferred_tokens_llama31_to_llama3.pt")
    transferer.save_transferred_embeddings(emb_path)

    del E_s, E_t, aligner
    gc.collect()
    logger.info("Freed embedding matrices and aligner before model integration")

    # --- Step 4: Integrate into target model ---
    logger.info("=== Step 4: Integrating transferred tokens into target model ===")
    t0 = time.time()
    transferer.integrate_into_model(TRANSFERRED_MODEL_DIR)
    compute_cost["model_integration_s"] = time.time() - t0
    logger.info(f"Model integration: {compute_cost['model_integration_s']:.1f}s")

    # --- Step 5: Verify saved model ---
    logger.info("=== Step 5: Verifying saved model ===")
    model_verify = transferer.verify_saved_model(TRANSFERRED_MODEL_DIR)
    all_stats["model_verification"] = model_verify
    assert model_verify["tokens_ok"], "Token encoding verification failed"
    assert model_verify["embeddings_ok"], "Embedding match verification failed"
    logger.info("Saved model verification PASSED")

    # --- Compute cost summary ---
    pipeline_total = time.time() - pipeline_start
    mem_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    compute_cost["pipeline_total_s"] = pipeline_total
    compute_cost["memory_peak_total_MB"] = mem_final[1] / 1024 / 1024
    compute_cost["comparison"] = {
        "transfer_method_s": compute_cost["alignment_svd_s"] + compute_cost["token_transfer_s"],
        "defensivetokens_optimization_gpu_hours": 16,
        "defensivetokens_optimization_description": "4 GPU-hours on 4xA100-80GB (=16 GPU-hours total)",
        "speedup_factor": f"{16 * 3600 / (compute_cost['alignment_svd_s'] + compute_cost['token_transfer_s']):.0f}x",
    }
    all_stats["compute_cost"] = compute_cost

    logger.info("=== Compute Cost Summary ===")
    logger.info(f"  Model loading: {compute_cost['model_loading_s']:.1f}s")
    logger.info(f"  Alignment SVD: {compute_cost['alignment_svd_s']:.2f}s")
    logger.info(f"  Token transfer: {compute_cost['token_transfer_s']:.4f}s")
    logger.info(f"  Model integration: {compute_cost['model_integration_s']:.1f}s")
    logger.info(f"  Pipeline total: {pipeline_total:.1f}s")
    logger.info(f"  Peak memory: {mem_final[1]/1024/1024:.0f} MB")
    logger.info(f"  vs DefensiveTokens optimization: {compute_cost['comparison']['defensivetokens_optimization_description']}")
    logger.info(f"  Speedup (alignment+transfer only): {compute_cost['comparison']['speedup_factor']}")

    stats_path = os.path.join(OUTPUT_DIR, "transfer_pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Pipeline stats saved to {stats_path}")

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
