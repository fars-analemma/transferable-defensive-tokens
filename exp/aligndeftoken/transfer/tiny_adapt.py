"""Tiny-adapt: fine-tune only the 5 transferred DefensiveToken embeddings using StruQ loss.
Single-GPU training with gradient checkpointing. BF16 8B model fits in 96GB.
Uses gradient masking to zero out gradients for all non-DT embedding rows.
Self-labels are mapped via orig_index to handle StruQ data shuffling.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from aligndeftoken.data.cleaned_alpaca import load_cleaned_alpaca, build_struq_dataset


class DefensiveTokenTrainingDataset(Dataset):

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        with open(data_path) as f:
            raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

        alpaca_data = load_cleaned_alpaca()
        struq_data = build_struq_dataset(alpaca_data, seed=42)

        output_map = {}
        for item in raw_data:
            output_map[item["index"]] = item.get("response", item.get("original_output", ""))

        self.samples = []
        replaced = 0
        for sample in struq_data:
            orig_idx = sample["orig_index"]
            if orig_idx in output_map and output_map[orig_idx].strip():
                sample["output"] = output_map[orig_idx]
                replaced += 1
            self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} training samples, replaced {replaced} labels with self-labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        conversation = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]

        prompt_msgs = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, add_defensive_tokens=True
        )
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, add_defensive_tokens=True
        )

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def save_checkpoint(model, tokenizer, dt_token_ids, output_dir, original_model_path):
    os.makedirs(output_dir, exist_ok=True)

    emb_weights = model.get_input_embeddings().weight.data.cpu()

    dt_embeddings = {}
    for i, tid in enumerate(dt_token_ids):
        dt_embeddings[f"token_{i}_id_{tid}"] = emb_weights[tid].float().numpy().tolist()
        logger.info(f"  DefensiveToken{i} (id={tid}): norm={emb_weights[tid].float().norm().item():.4f}")

    with open(os.path.join(output_dir, "dt_embeddings.json"), "w") as f:
        json.dump(dt_embeddings, f)

    base_model = AutoModelForCausalLM.from_pretrained(
        original_model_path, torch_dtype=torch.bfloat16
    )
    base_emb = base_model.get_input_embeddings()
    for tid in dt_token_ids:
        base_emb.weight.data[tid] = emb_weights[tid].to(base_emb.weight.dtype)

    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del base_model
    torch.cuda.empty_cache()

    logger.info(f"Checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=str, default="50,100,200")
    parser.add_argument("--wandb_project", type=str, default="transferable-defensive-tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_step_list = [int(s) for s in args.save_steps.split(",")]

    import wandb
    wandb.init(
        project=args.wandb_project,
        name=f"tiny-adapt-{os.path.basename(args.model_path)}",
        config=vars(args),
        mode="offline",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dt_token_ids = []
    for i in range(5):
        token_name = f"[DefensiveToken{i}]"
        ids = tokenizer.encode(token_name, add_special_tokens=False)
        assert len(ids) == 1, f"{token_name} should encode to 1 token, got {len(ids)}"
        dt_token_ids.append(ids[0])

    logger.info(f"DefensiveToken IDs: {dt_token_ids}")
    for tid in dt_token_ids:
        emb = model.get_input_embeddings().weight.data[tid]
        logger.info(f"  Token {tid}: norm={emb.float().norm().item():.4f}")

    for param in model.parameters():
        param.requires_grad = False

    emb_layer = model.get_input_embeddings()
    emb_layer.weight.requires_grad = True

    model.gradient_checkpointing_enable()
    model.cuda()

    dt_mask = torch.zeros(emb_layer.weight.shape[0], device="cuda", dtype=torch.bool)
    for tid in dt_token_ids:
        dt_mask[tid] = True

    optimizer = torch.optim.AdamW(
        [emb_layer.weight],
        lr=args.lr,
        weight_decay=0.0,
    )

    logger.info(f"Loading dataset from {args.data_path}")
    dataset = DefensiveTokenTrainingDataset(args.data_path, tokenizer, args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Starting training: {args.num_steps} steps, batch_size={args.batch_size}, "
                f"grad_accum={args.gradient_accumulation_steps}, lr={args.lr}")

    step = 0
    running_loss = 0.0
    log_steps = 0
    t0 = time.time()
    epoch = 0

    while step < args.num_steps:
        for batch in dataloader:
            if step >= args.num_steps:
                break

            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            with torch.no_grad():
                grad = emb_layer.weight.grad
                if grad is not None:
                    grad[~dt_mask] = 0.0

            running_loss += loss.item()
            log_steps += 1

            if log_steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                step += 1

                if step % 10 == 0:
                    elapsed = time.time() - t0
                    num_log_intervals = step // 10
                    avg_loss = running_loss / (10 * args.gradient_accumulation_steps) if step % 10 == 0 else running_loss / log_steps
                    logger.info(f"Step {step}/{args.num_steps}: loss={running_loss / (10 * args.gradient_accumulation_steps):.4f}, elapsed={elapsed:.1f}s")
                    wandb.log({"loss": running_loss / (10 * args.gradient_accumulation_steps), "step": step, "elapsed_s": elapsed})
                    running_loss = 0.0
                    log_steps = 0

                if step in save_step_list:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
                    save_checkpoint(model, tokenizer, dt_token_ids, ckpt_dir, args.model_path)
                    logger.info(f"Saved checkpoint at step {step} to {ckpt_dir}")

        epoch += 1
        logger.info(f"Epoch {epoch} complete, step={step}")

    final_dir = os.path.join(args.output_dir, "final")
    save_checkpoint(model, tokenizer, dt_token_ids, final_dir, args.model_path)
    logger.info(f"Saved final checkpoint to {final_dir}")

    wandb.finish()
    logger.info(f"Training complete. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
