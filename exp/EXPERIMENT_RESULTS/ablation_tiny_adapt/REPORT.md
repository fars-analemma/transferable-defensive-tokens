# Ablation: Tiny-Adapt from Procrustes vs Random Initialization

## Experiment Overview

Tests whether Procrustes-transferred DefensiveTokens provide a better initialization for efficient per-target fine-tuning compared to random initialization. The transfer direction is Llama-3 -> Llama-3.1 (the "hard" direction where Procrustes-only transfer achieves only partial success with gap-closed r=0.52).

Both conditions use identical training: 200 gradient steps updating only the 5 DefensiveToken embeddings (20,480 parameters) with StruQ loss on self-labeled Cleaned Alpaca data.

## Setup

- **Target model**: meta-llama/Llama-3.1-8B-Instruct
- **Source model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Training**: 200 steps, LR=0.1, batch_size=4, grad_accum=4 (effective batch 16), AdamW
- **Data**: 51,760 Cleaned Alpaca samples with StruQ protocol, self-labeled by undefended target model
- **Evaluation**: 208 AlpacaFarm injection prompts x 3 attack variants + 208 benign prompts
- **Checkpoints**: Saved at steps 25, 50, 100, 150, 200; best checkpoint selected by lowest max ASR
- **Compute**: 1 GPU per run, ~7 min training + ~30 min evaluation per condition

### Initialization Conditions
- **Procrustes Init**: Start from Procrustes-transferred DefensiveToken embeddings (ASR=33.7% before training)
- **Random Init**: Start from N(0,I) random embeddings (ASR=69.2% = No Defense, before training)

## Key Results

| Init Method | Best Step | Max ASR | RefusalRate | GPU-Hours |
|-------------|-----------|---------|-------------|-----------|
| Procrustes  | 150       | **1.9%**| 1.0%        | 0.175h    |
| Random      | 100       | 2.9%    | 0.5%        | 0.117h    |
| Full DT (measured) | ~3200 | 1.9% | 1.0%       | ~16h      |
| No Defense  | N/A       | 69.2%   | 0.5%        | 0h        |

### Per-Step Results

| Steps | Procrustes ASR | Random ASR |
|-------|----------------|------------|
| 0     | 33.7%          | 69.2%      |
| 25    | 2.4%           | 8.2%       |
| 50    | 4.8%           | 11.5%      |
| 100   | 2.4%           | 2.9%       |
| 150   | 1.9%           | 4.8%       |
| 200   | 1.9%           | 7.2%       |

### Speedup Analysis

| ASR Threshold | Procrustes Steps | Random Steps | Speedup |
|---------------|-----------------|--------------|---------|
| < 5%          | 25              | 100          | **4.0x**|
| < 3%          | 25              | 100          | **4.0x**|
| < 2%          | 150             | Never        | **Inf** |

## Key Observations

1. **Procrustes init converges faster**: Reaches <5% ASR in just 25 steps (vs 100 steps for random init), a 4x speedup.

2. **Procrustes init reaches lower ASR**: Best checkpoint achieves 1.9% ASR (matching Full DT), while random init's best is 2.9% and never reaches the 2% threshold within 200 steps.

3. **Random init is unstable**: ASR oscillates (8.2% -> 11.5% -> 2.9% -> 4.8% -> 7.2%), while Procrustes init is more stable after the initial drop (2.4% -> 4.8% -> 2.4% -> 1.9% -> 1.9%).

4. **Both dramatically faster than full training**: Even the random-init control achieves 2.9% ASR in 100 steps (~0.12 GPU-hours), compared to ~16 GPU-hours for full DefensiveTokens optimization. The StruQ objective with self-labels is highly sample-efficient.

5. **Procrustes provides meaningful structure**: The 4x speedup and superior final ASR confirm that Procrustes alignment transfers meaningful defensive information, not just noise. The transferred embeddings are close enough to the target optimum that a small number of gradient steps suffice.
