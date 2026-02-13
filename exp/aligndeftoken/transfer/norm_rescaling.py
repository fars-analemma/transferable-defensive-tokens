"""Norm rescaling variants for transferred DefensiveToken embeddings.

Two modes:
  (a) Source-norm rescaling: match ||T_s[i]||_2  (no-op for orthogonal W*)
  (b) Target-native-norm rescaling: match ||T_target_native[i]||_2

Used in ablation to test whether absolute norm magnitude matters for defense.
"""

import json
import logging
import os

import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_LLAMA3 = """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n"""


def source_norm_rescale(T_t: np.ndarray, T_s: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rescale each T_t[i] to match ||T_s[i]||_2.

    For orthogonal Procrustes (W^T W = I), this is a no-op because
    ||T_s @ W||_2 = ||T_s||_2. We verify this numerically.
    """
    source_norms = np.linalg.norm(T_s, axis=1)
    transferred_norms = np.linalg.norm(T_t, axis=1)
    scale_factors = source_norms / transferred_norms

    T_rescaled = T_t * scale_factors[:, np.newaxis]
    rescaled_norms = np.linalg.norm(T_rescaled, axis=1)

    norm_diffs = np.abs(transferred_norms - source_norms)
    is_noop = bool(np.all(norm_diffs < 1e-3))

    stats = {
        "mode": "source_norm",
        "source_norms": source_norms.tolist(),
        "transferred_norms": transferred_norms.tolist(),
        "rescaled_norms": rescaled_norms.tolist(),
        "scale_factors": scale_factors.tolist(),
        "norm_diffs_before_rescale": norm_diffs.tolist(),
        "max_norm_diff": float(np.max(norm_diffs)),
        "is_noop": is_noop,
    }

    logger.info(f"Source-norm rescaling: max_norm_diff={np.max(norm_diffs):.2e}, is_noop={is_noop}")
    for i in range(len(T_t)):
        logger.info(
            f"  Token {i}: ||T_s||={source_norms[i]:.4f}, ||T_t||={transferred_norms[i]:.4f}, "
            f"scale={scale_factors[i]:.6f}, ||rescaled||={rescaled_norms[i]:.4f}"
        )

    return T_rescaled, stats


def target_native_norm_rescale(T_t: np.ndarray, T_native: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rescale each T_t[i] to match ||T_target_native[i]||_2."""
    native_norms = np.linalg.norm(T_native, axis=1)
    transferred_norms = np.linalg.norm(T_t, axis=1)
    scale_factors = native_norms / transferred_norms

    T_rescaled = T_t * scale_factors[:, np.newaxis]
    rescaled_norms = np.linalg.norm(T_rescaled, axis=1)

    stats = {
        "mode": "target_native_norm",
        "native_target_norms": native_norms.tolist(),
        "transferred_norms": transferred_norms.tolist(),
        "rescaled_norms": rescaled_norms.tolist(),
        "scale_factors": scale_factors.tolist(),
        "norm_ratio_native_over_transferred": (native_norms / transferred_norms).tolist(),
    }

    logger.info("Target-native-norm rescaling:")
    for i in range(len(T_t)):
        logger.info(
            f"  Token {i}: ||T_native||={native_norms[i]:.4f}, ||T_t||={transferred_norms[i]:.4f}, "
            f"scale={scale_factors[i]:.6f}, ||rescaled||={rescaled_norms[i]:.4f}"
        )

    return T_rescaled, stats


def integrate_rescaled_into_model(
    T_rescaled: np.ndarray,
    target_model_name: str,
    output_dir: str,
) -> str:
    """Save rescaled tokens into a copy of the target model (same as token_transfer.py pattern)."""
    logger.info(f"Loading target model: {target_model_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_name)

    model_dtype = model.get_input_embeddings().weight.dtype
    T_tensor = torch.tensor(T_rescaled, dtype=torch.float32).to(model_dtype).to(model.device)
    num_tokens = T_tensor.shape[0]

    special_token_names = [f"[DefensiveToken{i}]" for i in range(num_tokens)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_token_names})

    model.resize_token_embeddings(len(tokenizer))
    for i in range(num_tokens):
        model.get_input_embeddings().weight.data[-num_tokens + i] = T_tensor[i]

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.chat_template = CHAT_TEMPLATE_LLAMA3
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved rescaled model to {output_dir}")

    del model
    return output_dir
