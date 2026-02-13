# Optimization Iteration 0: Tiny-Adapt for Reverse Transfer (Llama-3 -> Llama-3.1)

## Experiment Overview

The Procrustes-only reverse transfer (Llama-3 -> Llama-3.1) achieved only 33.7% max ASR (gap-closed ratio 0.517), far below the 0.85 threshold. This optimization applies "tiny-adapt": fine-tuning only the 5 transferred DefensiveToken embeddings (20,480 parameters) using the StruQ defensive training loss, while keeping the entire 8B model frozen.

## Issues Diagnosed and Fixed

### Issue 1: Data Mapping Bug (Critical)
The self-label indices (from raw Alpaca dataset order) were incorrectly mapped to StruQ training samples (which are shuffled). Fixed by adding `orig_index` tracking in `build_struq_dataset()` and using it for self-label lookup in `tiny_adapt.py`.

### Issue 2: FSDP Complexity (Reliability)
Removed unnecessary multi-GPU FSDP setup. The 8B BF16 model with gradient checkpointing fits on a single 96GB GPU. This eliminates potential gradient masking issues with FSDP sharded parameters.

### Issue 3: Loss Logging Bug (Minor)
Fixed incorrect loss averaging formula in the training loop.

## Setup

- **Model**: Llama-3.1-8B-Instruct with 5 Procrustes-transferred DefensiveTokens from Llama-3
- **Training data**: 51,760 Cleaned Alpaca samples with StruQ protocol (50% benign, 25% ignore-attack, 25% completion-attack)
- **Labels**: Self-labeled by undefended Llama-3.1-8B-Instruct (per DefensiveTokens paper recipe)
- **Optimizer**: AdamW, LR=0.1, no weight decay
- **Batch**: 4 per GPU, gradient accumulation 4 (effective batch 16)
- **Steps**: 200 (saved checkpoints at 25, 50, 100, 150, 200)
- **GPUs**: 1x 96GB GPU
- **Training time**: ~7 minutes
- **Self-label generation**: ~40 minutes via vLLM on 1 GPU
- **WandB**: offline logging

## Key Results

| Checkpoint | Max ASR (%) | Gap-Closed (pub) | Refusal Rate (%) |
|------------|-------------|-------------------|------------------|
| Baseline   | 33.7        | 0.517             | 1.0              |
| step_25    | 2.4         | 0.972             | 1.0              |
| step_50    | 4.8         | 0.937             | 0.0              |
| step_100   | 2.4         | 0.972             | 0.0              |
| step_150   | 1.9         | 0.979             | 0.5              |
| **step_200** | **1.9**   | **0.979**         | **0.0**          |

Best: step_200/final with **1.9% max ASR** and **0.0% refusal rate**.

### Per-Attack-Type ASR (step_200)
- ignore: 1.9% (4/208)
- completion: 1.0% (2/208)
- ignore_completion: 0.0% (0/208)

### Gap-Closed Ratios (step_200)
- vs. published full DT (0.5% ASR): **0.979**
- vs. measured full DT (1.9% ASR): **1.000**

## Key Observations

1. The tiny-adapt method is extremely effective: even 25 gradient steps reduce ASR from 33.7% to 2.4%.
2. The best result (1.9% max ASR) exactly matches the measured full DT performance, meaning the transferred + adapted tokens perform identically to natively-trained ones.
3. Training is very fast (~7 minutes on 1 GPU) since only 20,480 parameters are updated.
4. The loss curve shows steady convergence from 0.074 to ~0.05 over 200 steps.
5. Token embedding norms grow from ~80-87 (initial transfer) to ~94-106 (step 200), indicating the model is learning stronger defensive representations.

## Training Jobs
- Self-label generation: `dlcpeb66hyj8mdwd` (Succeeded)
- Tiny-adapt training: `dlcpyoemhxwcn3zo` (Succeeded)
- Checkpoint evaluation: `dlc7djtopmj1kja9` (evaluation succeeded, comparison script had minor formatting issue)
