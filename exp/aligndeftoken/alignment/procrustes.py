"""Orthogonal Procrustes alignment between LLM embedding spaces.

Computes W* = argmin_{W^T W = I} ||XW - Y||_F using SVD of X^T Y.
Both models must share the same tokenizer (same vocab, same token ordering).
Used to map DefensiveToken embeddings from source to target model space.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)


class ProcrustesAligner:
    def __init__(self, top_k_tokens: Optional[int] = None):
        self.top_k_tokens = top_k_tokens
        self.W = None
        self.disparity = None
        self.stats = {}

    def extract_embeddings(self, model) -> np.ndarray:
        emb = model.get_input_embeddings().weight.data.float().cpu().numpy()
        logger.info(f"Extracted embedding matrix: shape={emb.shape}, dtype={emb.dtype}")
        return emb

    def _select_token_indices(self, E_s: np.ndarray, E_t: np.ndarray) -> np.ndarray:
        vocab_size = E_s.shape[0]
        if self.top_k_tokens is not None and self.top_k_tokens < vocab_size:
            norms_s = np.linalg.norm(E_s, axis=1)
            norms_t = np.linalg.norm(E_t, axis=1)
            avg_norms = (norms_s + norms_t) / 2
            indices = np.argsort(-avg_norms)[:self.top_k_tokens]
            logger.info(f"Using top-{self.top_k_tokens} tokens by average norm (of {vocab_size})")
            return np.sort(indices)
        logger.info(f"Using all {vocab_size} tokens for alignment")
        return np.arange(vocab_size)

    def fit(self, E_s: np.ndarray, E_t: np.ndarray) -> dict:
        t0 = time.time()

        indices = self._select_token_indices(E_s, E_t)
        X = E_s[indices].astype(np.float64)
        Y = E_t[indices].astype(np.float64)

        logger.info(f"Solving Procrustes: X={X.shape}, Y={Y.shape}, dtype={X.dtype}")
        W, scale = orthogonal_procrustes(X, Y)
        elapsed = time.time() - t0

        self.W = W
        self.disparity = scale

        orth_err = np.linalg.norm(W.T @ W - np.eye(W.shape[0]))
        residual = np.linalg.norm(X @ W - Y, "fro")

        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(indices), size=min(100, len(indices)), replace=False)
        per_token_err = np.linalg.norm(X[sample_idx] @ W - Y[sample_idx], axis=1)
        mean_err = float(np.mean(per_token_err))
        max_err = float(np.max(per_token_err))

        self.stats = {
            "num_tokens_used": len(indices),
            "disparity_scale": float(scale),
            "residual_frobenius": float(residual),
            "orthogonality_error": float(orth_err),
            "sample_alignment_mean_err": mean_err,
            "sample_alignment_max_err": max_err,
            "alignment_time_s": elapsed,
            "W_shape": list(W.shape),
        }

        logger.info(f"Procrustes alignment complete in {elapsed:.2f}s")
        logger.info(f"  Orthogonality error ||W^T W - I||_F = {orth_err:.2e}")
        logger.info(f"  Disparity (scale) = {scale:.6f}")
        logger.info(f"  Residual ||XW - Y||_F = {residual:.4f}")
        logger.info(f"  Sample per-token error: mean={mean_err:.4f}, max={max_err:.4f}")

        return self.stats

    def transform(self, T_s: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Must call fit() before transform()")
        return T_s @ self.W

    def get_alignment_matrix(self) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Must call fit() before get_alignment_matrix()")
        return self.W

    def save(self, path: str):
        if self.W is None:
            raise RuntimeError("Must call fit() before save()")
        W_tensor = torch.from_numpy(self.W)
        torch.save({"W": W_tensor, "stats": self.stats}, path)
        logger.info(f"Saved alignment matrix to {path}")

    @classmethod
    def load(cls, path: str) -> "ProcrustesAligner":
        data = torch.load(path, weights_only=False)
        aligner = cls()
        aligner.W = data["W"].numpy()
        aligner.stats = data.get("stats", {})
        aligner.disparity = aligner.stats.get("disparity_scale")
        logger.info(f"Loaded alignment matrix from {path}: shape={aligner.W.shape}")
        return aligner
