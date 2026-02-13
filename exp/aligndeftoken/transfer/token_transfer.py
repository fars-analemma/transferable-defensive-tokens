"""Transfer DefensiveToken embeddings from source to target model using Procrustes W*.

Loads pre-optimized tokens T_s from defensivetokens.json, applies T_t = T_s @ W*,
and integrates T_t into the target model's vocabulary (same pattern as DefensiveToken/setup.py).
"""

import json
import logging
import os
from typing import Optional

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


class TransferDefensiveTokens:
    def __init__(
        self,
        defensivetokens_path: str,
        source_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        target_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        self.defensivetokens_path = defensivetokens_path
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        self.T_s = None
        self.T_t = None
        self.native_target_tokens = None

    def load_source_tokens(self) -> np.ndarray:
        with open(self.defensivetokens_path) as f:
            dt = json.load(f)
        self.T_s = np.array(dt[self.source_model_name], dtype=np.float32)
        logger.info(f"Loaded source DefensiveTokens: shape={self.T_s.shape} from {self.source_model_name}")

        if self.target_model_name in dt:
            self.native_target_tokens = np.array(dt[self.target_model_name], dtype=np.float32)
            logger.info(f"Loaded native target DefensiveTokens for comparison: shape={self.native_target_tokens.shape}")

        return self.T_s

    def transfer(self, W: np.ndarray) -> np.ndarray:
        if self.T_s is None:
            raise RuntimeError("Must call load_source_tokens() first")
        self.T_t = self.T_s @ W
        logger.info(f"Transferred tokens: T_t shape={self.T_t.shape}")
        return self.T_t

    def verify_transfer(self) -> dict:
        if self.T_s is None or self.T_t is None:
            raise RuntimeError("Must call load_source_tokens() and transfer() first")

        source_norms = np.linalg.norm(self.T_s, axis=1)
        transferred_norms = np.linalg.norm(self.T_t, axis=1)
        norm_diff = np.abs(source_norms - transferred_norms)

        stats = {
            "source_norms": source_norms.tolist(),
            "transferred_norms": transferred_norms.tolist(),
            "norm_diff_max": float(np.max(norm_diff)),
            "norm_preservation_ok": bool(np.max(norm_diff) < 1e-3),
        }

        logger.info("Token norm comparison (source vs transferred):")
        for i in range(len(source_norms)):
            logger.info(f"  Token {i}: ||T_s||={source_norms[i]:.4f}, ||T_t||={transferred_norms[i]:.4f}, diff={norm_diff[i]:.2e}")

        if self.native_target_tokens is not None:
            native_norms = np.linalg.norm(self.native_target_tokens, axis=1)
            stats["native_target_norms"] = native_norms.tolist()
            cosine_sims = []
            for i in range(len(self.T_t)):
                cos = np.dot(self.T_t[i], self.native_target_tokens[i]) / (transferred_norms[i] * native_norms[i])
                cosine_sims.append(float(cos))
            stats["cosine_sim_to_native"] = cosine_sims
            logger.info("Comparison to native target DefensiveTokens:")
            for i in range(len(native_norms)):
                logger.info(f"  Token {i}: ||native||={native_norms[i]:.4f}, ||transferred||={transferred_norms[i]:.4f}, cos_sim={cosine_sims[i]:.4f}")

        return stats

    def integrate_into_model(self, output_dir: str) -> str:
        if self.T_t is None:
            raise RuntimeError("Must call transfer() first")

        logger.info(f"Loading target model: {self.target_model_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.target_model_name, torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.target_model_name)

        model_dtype = model.get_input_embeddings().weight.dtype
        T_t_tensor = torch.tensor(self.T_t, dtype=torch.float32).to(model_dtype).to(model.device)
        num_tokens = T_t_tensor.shape[0]

        special_token_names = [f"[DefensiveToken{i}]" for i in range(num_tokens)]
        tokenizer.add_special_tokens({"additional_special_tokens": special_token_names})

        model.resize_token_embeddings(len(tokenizer))
        for i in range(num_tokens):
            model.get_input_embeddings().weight.data[-num_tokens + i] = T_t_tensor[i]

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.chat_template = CHAT_TEMPLATE_LLAMA3
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved transferred model to {output_dir}")

        return output_dir

    def verify_saved_model(self, output_dir: str) -> dict:
        if self.T_t is None:
            raise RuntimeError("Must call transfer() first")

        logger.info(f"Verifying saved model at {output_dir}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16
        )

        results = {"tokens_ok": True, "embeddings_ok": True, "details": []}
        for i in range(self.T_t.shape[0]):
            token_name = f"[DefensiveToken{i}]"
            token_ids = tokenizer.encode(token_name, add_special_tokens=False)

            if len(token_ids) != 1:
                logger.error(f"{token_name} encodes to {len(token_ids)} tokens: {token_ids}")
                results["tokens_ok"] = False
                continue

            token_id = token_ids[0]
            emb = model.get_input_embeddings().weight.data[token_id].float().cpu().numpy()
            expected = self.T_t[i]
            diff = np.max(np.abs(emb - expected))

            ok = diff < 0.1
            if not ok:
                results["embeddings_ok"] = False

            results["details"].append({
                "token": token_name,
                "token_id": token_id,
                "max_abs_diff": float(diff),
                "match": ok,
            })
            logger.info(f"  {token_name}: id={token_id}, max_diff={diff:.2e}, {'OK' if ok else 'MISMATCH'}")

        del model
        return results

    def save_transferred_embeddings(self, path: str):
        if self.T_t is None:
            raise RuntimeError("Must call transfer() first")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(torch.from_numpy(self.T_t), path)
        logger.info(f"Saved transferred embeddings to {path}: shape={self.T_t.shape}")
