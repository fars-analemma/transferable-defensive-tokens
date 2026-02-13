"""Create a random-initialized DefensiveTokens model for Llama-3.1-8B-Instruct.
Copies the transferred-tokens model (which has the expanded vocabulary with 5 DT token IDs),
then replaces the 5 DT embedding rows with N(0,I) random vectors.
Includes verification that the model is correctly set up.
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE_MODEL = os.path.join(BASE, "DefensiveToken/meta-llama/Llama-3.1-8B-Instruct-5TransferredTokens")
OUTPUT_MODEL = os.path.join(BASE, "DefensiveToken/meta-llama/Llama-3.1-8B-Instruct-5RandomTokens")
TRANSFERRED_TOKENS_PATH = os.path.join(BASE, "aligndeftoken/outputs/transferred_tokens_llama3_to_llama31.pt")

SEED = 42
NUM_DT = 5
HIDDEN_DIM = 4096


def main():
    print(f"Loading source model: {SOURCE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)

    dt_token_ids = []
    for i in range(NUM_DT):
        token_name = f"[DefensiveToken{i}]"
        ids = tokenizer.encode(token_name, add_special_tokens=False)
        assert len(ids) == 1, f"{token_name} should encode to 1 token, got {len(ids)}: {ids}"
        dt_token_ids.append(ids[0])
    print(f"DefensiveToken IDs: {dt_token_ids}")

    emb = model.get_input_embeddings()
    print(f"Embedding shape: {emb.weight.shape}")

    print("\nBefore replacement (transferred token norms):")
    for i, tid in enumerate(dt_token_ids):
        norm = emb.weight.data[tid].float().norm().item()
        print(f"  DT{i} (id={tid}): norm={norm:.4f}")

    torch.manual_seed(SEED)
    random_embeddings = torch.randn(NUM_DT, HIDDEN_DIM, dtype=torch.float32)

    for i, tid in enumerate(dt_token_ids):
        emb.weight.data[tid] = random_embeddings[i].to(emb.weight.dtype)

    print("\nAfter replacement (random token norms):")
    for i, tid in enumerate(dt_token_ids):
        norm = emb.weight.data[tid].float().norm().item()
        print(f"  DT{i} (id={tid}): norm={norm:.4f}")

    os.makedirs(OUTPUT_MODEL, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL)
    tokenizer.save_pretrained(OUTPUT_MODEL)
    print(f"\nSaved random-init model to {OUTPUT_MODEL}")

    verify(OUTPUT_MODEL, dt_token_ids, random_embeddings)


def verify(model_path, dt_token_ids, expected_random_embs):
    print("\n=== VERIFICATION ===")

    print("1. Reloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    emb = model.get_input_embeddings()

    print("2. Checking tokenizer encodes DT tokens as single tokens...")
    for i in range(NUM_DT):
        token_name = f"[DefensiveToken{i}]"
        ids = tokenizer.encode(token_name, add_special_tokens=False)
        assert len(ids) == 1, f"FAIL: {token_name} encodes to {len(ids)} tokens"
        assert ids[0] == dt_token_ids[i], f"FAIL: {token_name} id mismatch: {ids[0]} != {dt_token_ids[i]}"
    print("   PASS: All DT tokens encode as single tokens with correct IDs")

    print("3. Checking DT embedding norms (expect ~sqrt(4096) ≈ 64)...")
    for i, tid in enumerate(dt_token_ids):
        norm = emb.weight.data[tid].float().norm().item()
        expected_range = (40.0, 90.0)
        assert expected_range[0] < norm < expected_range[1], \
            f"FAIL: DT{i} norm={norm:.2f} outside expected range {expected_range}"
        print(f"   DT{i}: norm={norm:.4f} (OK)")
    print("   PASS: All norms in expected range")

    print("4. Checking DT embeddings match expected random values...")
    for i, tid in enumerate(dt_token_ids):
        actual = emb.weight.data[tid].float()
        expected = expected_random_embs[i].float()
        cosine = torch.nn.functional.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0)).item()
        assert cosine > 0.99, f"FAIL: DT{i} cosine with expected random = {cosine:.4f}"
    print("   PASS: All DT embeddings match expected random initialization")

    print("5. Checking DT embeddings differ from transferred tokens...")
    transferred = torch.load(TRANSFERRED_TOKENS_PATH, map_location="cpu", weights_only=True)
    if isinstance(transferred, dict):
        transferred = transferred.get("transferred_tokens", transferred)
    if isinstance(transferred, torch.Tensor) and transferred.shape == (NUM_DT, HIDDEN_DIM):
        for i, tid in enumerate(dt_token_ids):
            actual = emb.weight.data[tid].float()
            trans = transferred[i].float()
            cosine = torch.nn.functional.cosine_similarity(actual.unsqueeze(0), trans.unsqueeze(0)).item()
            print(f"   DT{i}: cosine(random, transferred) = {cosine:.4f}")
            assert abs(cosine) < 0.5, f"FAIL: DT{i} too similar to transferred tokens (cosine={cosine:.4f})"
        print("   PASS: Random embeddings differ from transferred tokens")
    else:
        print(f"   SKIP: Transferred tokens file has unexpected format: {type(transferred)}")

    print("\n=== ALL VERIFICATION CHECKS PASSED ===")
    del model


if __name__ == "__main__":
    main()
