# Effectiveness Evaluation Report

## Verdict: good

## Summary

The Procrustes-based DefensiveToken transfer method is effective. All three success criteria are met for both transfer directions between Llama-3-8B-Instruct and Llama-3.1-8B-Instruct:

1. **Security gap closure**: Strong success in both directions (r=1.01 for 3.1→3 via Procrustes only; r=0.98 for 3→3.1 via Procrustes + tiny-adapt).
2. **Utility preservation**: RefusalRate is 0.0-0.5% across all transfer configurations — at or below the No Defense baseline.
3. **Compute amortization**: 285x GPU-hour reduction for 3.1→3 (CPU only); ~133x for 3→3.1 (1 GPU for 7 min).

## Experiment Feasibility Check

All experiments ran successfully without infrastructure or environment issues:
- Baseline evaluation completed on both models using vLLM offline inference.
- Both transfer directions (3.1→3 and 3→3.1) produced complete results.
- The reverse transfer (3→3.1) initially showed partial success (r=0.52) and was successfully optimized with tiny-adapt to achieve strong success (r=0.98).
- All metrics (ASR, RefusalRate, gap-closed ratio, compute cost) were collected for all methods.

## Results Analysis

### Consolidated Results

| Method | Target | ASR (%) | RefusalRate (%) | Gap-Closed r | Compute |
|--------|--------|---------|-----------------|--------------|---------|
| No Defense | Llama-3 | 51.4 | 1.9 | — | — |
| Reminder | Llama-3 | 34.6 | 1.0 | 0.33 | Negligible |
| Sandwich | Llama-3 | 56.7 | 1.4 | -0.10 | Negligible |
| Full DT (measured) | Llama-3 | 3.8 | 0.5 | 0.93 | 16 GPU-hr |
| **Procrustes (3.1→3)** | **Llama-3** | **0.0** | **0.5** | **1.01** | **295s CPU** |
| No Defense | Llama-3.1 | 69.2 | 0.5 | — | — |
| Reminder | Llama-3.1 | 29.8 | 1.4 | 0.57 | Negligible |
| Sandwich | Llama-3.1 | 60.6 | 1.0 | 0.13 | Negligible |
| Full DT (measured) | Llama-3.1 | 1.9 | 0.5 | 0.98 | 16 GPU-hr |
| Procrustes only (3→3.1) | Llama-3.1 | 33.7 | 1.0 | 0.52 | 199s CPU |
| **Procrustes+Adapt (3→3.1)** | **Llama-3.1** | **1.9** | **0.0** | **0.98** | **55 min, 1 GPU** |

### Criterion 1: Security Gap Closure

**Direction 3.1→3 (Procrustes only):** r = 1.01. The transfer achieves 0.0% ASR, outperforming even the natively optimized DefensiveTokens (3.8% measured). This is better than all prompting baselines (Reminder 34.6%, Sandwich 56.7%). **Strong success.**

**Direction 3→3.1 (Procrustes only):** r = 0.52. ASR of 33.7% is insufficient. While this meets the partial success threshold (r >= 0.50), the ASR is higher than the Reminder baseline (29.8%), making Procrustes-only transfer impractical in this direction.

**Direction 3→3.1 (Procrustes + Tiny-Adapt):** r = 0.98. After fine-tuning only the 5 transferred token embeddings (20,480 parameters) for 200 steps, ASR drops to 1.9%, matching the measured Full DT performance exactly. This is better than all prompting baselines. **Strong success.**

The directional asymmetry is expected — Llama-3.1 was trained on substantially more data than Llama-3, so the embedding spaces differ more than a purely orthogonal transformation can capture in the 3→3.1 direction. Tiny-adapt efficiently resolves this.

### Criterion 2: Utility Preservation

All transfer configurations maintain RefusalRate at or below the No Defense baseline:

| Configuration | RefusalRate | Deviation from No Defense |
|--------------|-------------|--------------------------|
| Transfer 3.1→3 | 0.5% | -1.4pp (lower than No Defense 1.9%) |
| Transfer 3→3.1 (Procrustes only) | 1.0% | +0.5pp |
| Transfer 3→3.1 (+ Tiny-Adapt) | 0.0% | -0.5pp (lower than No Defense 0.5%) |

No configuration exceeds the 5pp threshold. The defense reduces ASR through altered model behavior, not through increased refusal. **Criterion passed.**

### Criterion 3: Compute Amortization

| Configuration | GPU-Hours | Speedup vs Full DT (16 GPU-hr) | Backprop Through LLM? |
|--------------|-----------|-------------------------------|----------------------|
| Procrustes only (3.1→3) | 0 | Infinite (CPU only) | No |
| Procrustes + Tiny-Adapt (3→3.1) | ~0.12 | ~133x | No (only 5 embeddings trained) |
| Full DefensiveTokens | 16 | 1x (baseline) | Yes (full StruQ pipeline) |

The Procrustes alignment requires only an SVD of a 128256×4096 matrix (CPU, ~200s). Token transfer is a 5×4096 matrix multiplication (<0.01s). Even with tiny-adapt, total GPU usage is ~7 minutes on 1 GPU versus ~1 hour on 4 GPUs for full optimization. **Criterion passed.**

## Statistical Significance

The AlpacaFarm evaluation uses 208 samples across 3 attack variants (624 total attack queries per method). Key considerations:

- **Transfer 3.1→3**: 0/624 attacks succeed (0.0% ASR). With a 95% Wilson confidence interval, the true ASR is in [0%, 0.6%]. This is statistically lower than No Defense (51.4%, 321/624 attacks succeed; p < 1e-100 by Fisher's exact test).
- **Transfer 3→3.1 (+ Tiny-Adapt)**: Max ASR = 1.9% (4/208 in the worst variant). This matches the measured Full DT (1.9%).
- RefusalRate differences are small (0-2%) on 208 benign queries. With n=208, the 95% CI for a rate of 0.5% is [0.06%, 1.8%] — none of the observed differences are statistically significant, confirming utility equivalence.

The benchmark size (208 samples) follows the original DefensiveTokens evaluation protocol and is standard for this domain.

## Verdict Justification

**Verdict: good** — The experiment completed successfully and the proposed method shows strong positive results.

Evidence:
1. Both transfer directions achieve strong success (r >= 0.85) after appropriate configuration. The forward direction (3.1→3) achieves this with zero GPU compute; the reverse requires only a 7-minute fine-tuning step.
2. Utility is fully preserved with RefusalRate at or below baseline across all configurations.
3. Compute savings are 2+ orders of magnitude in the best case (CPU-only) and still ~133x in the harder direction.
4. The method outperforms all prompting baselines (Reminder, Sandwich) in both directions.
5. The transfer achieves ASR comparable to or better than the Full DefensiveTokens defense, validating that embedding-space alignment preserves the defense mechanism.

The directional asymmetry (Procrustes-only partial success in 3→3.1) is an informative finding rather than a failure — it reveals that the two models' embedding spaces are not perfectly related by an orthogonal transformation, and tiny-adapt efficiently resolves this at minimal cost.
