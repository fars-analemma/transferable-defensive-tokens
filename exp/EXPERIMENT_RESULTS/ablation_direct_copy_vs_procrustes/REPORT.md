# Ablation: Direct Copy vs Procrustes Transfer

## Experiment Overview

This ablation study isolates the contribution of Procrustes embedding-space alignment by comparing:
- **Direct Copy**: Use source model's DefensiveToken embeddings T_s directly on the target model (T_t = T_s, no transformation)
- **Procrustes Transfer**: Apply orthogonal Procrustes alignment before transferring (T_t = T_s @ W*)

Both methods are evaluated on both transfer directions using the AlpacaFarm benchmark (208 samples, 3 attack variants).

## Setup

- Source/Target models: Llama-3-8B-Instruct <-> Llama-3.1-8B-Instruct
- Benchmark: AlpacaFarm (208 samples with input field)
- Attack variants: ignore, completion, ignore_completion (report max ASR)
- Metrics: ASR (attack success rate, lower is better), RefusalRate (lower is better), Gap-Closed r (higher is better)
- Gap-Closed r = (ASR_no_defense - ASR_method) / (ASR_no_defense - ASR_full_dt)

## Key Results

### Direction: Llama-3.1 -> Llama-3

| Method | ASR (%) | RefusalRate (%) | Gap-Closed r |
|--------|---------|-----------------|--------------|
| No Defense | 51.4 | 1.9 | 0.0 |
| Direct Copy | 0.0 | 0.5 | 1.08 |
| Procrustes Transfer | 0.0 | 0.5 | 1.08 |
| Full DefensiveTokens | 3.8 | 0.5 | 1.0 |

**Procrustes improvement over Direct Copy: 0.0 pp ASR reduction**

### Direction: Llama-3 -> Llama-3.1

| Method | ASR (%) | RefusalRate (%) | Gap-Closed r |
|--------|---------|-----------------|--------------|
| No Defense | 69.2 | 0.5 | 0.0 |
| Direct Copy | 34.6 | 1.0 | 0.51 |
| Procrustes Transfer | 33.7 | 1.0 | 0.53 |
| Full DefensiveTokens | 1.9 | 0.5 | 1.0 |

**Procrustes improvement over Direct Copy: 0.9 pp ASR reduction (gap-closed +0.013)**

## Key Observations

1. **Direction 3.1->3: Alignment is unnecessary.** Direct Copy achieves identical performance to Procrustes Transfer (both 0.0% ASR). This confirms the embedding spaces of Llama-3 and Llama-3.1 are nearly identical in the 3.1->3 direction, and the rotation learned by Procrustes is close to identity.

2. **Direction 3->3.1: Alignment provides marginal benefit.** Procrustes reduces ASR by only 0.9 percentage points compared to Direct Copy (33.7% vs 34.6%). The improvement is minimal, suggesting the rotation is small but not exactly identity. Both methods still leave a large gap to Full DefensiveTokens (1.9% ASR), indicating that token transfer alone (with or without alignment) is insufficient in this direction -- the directional asymmetry is the dominant challenge, not the embedding rotation.

3. **Directional asymmetry dominates.** The performance difference between 3.1->3 (0% ASR) and 3->3.1 (34% ASR) is far larger than the difference between Direct Copy and Procrustes within each direction. This confirms the core finding from earlier tasks: the two models' embedding spaces are nearly identical in rotation, but differ in how well the defensive tokens generalize across the version gap.

4. **Procrustes alignment is a safe no-op.** Since Procrustes never hurts (identical or slightly better), it remains the recommended approach as a general-purpose transfer method, even when direct copy would suffice.
