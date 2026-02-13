# Ablation: Norm Rescaling for Transferred DefensiveTokens

## Experiment Overview

This ablation tests whether rescaling transferred DefensiveToken embeddings to match specific L2 norm targets improves transfer quality. Orthogonal Procrustes preserves vector norms by construction (W* is orthogonal, so ||T_t||_2 = ||T_s||_2). We verify this numerically and test whether adjusting norms to match the target model's natively-optimized DefensiveToken norms has any effect.

## Setup

- **Source/Target models**: Llama-3-8B-Instruct <-> Llama-3.1-8B-Instruct
- **Evaluation**: AlpacaFarm (208 samples, 3 attack variants), ASR + RefusalRate
- **Conditions**:
  - (a) Procrustes transfer (no rescaling) -- reused from previous experiments
  - (b) Source-norm rescaling: T_t[i] * (||T_s[i]|| / ||T_t[i]||)
  - (c) Target-native-norm rescaling: T_t[i] * (||T_native[i]|| / ||T_t[i]||)

## Key Results

### Norm Preservation Verification

Orthogonal Procrustes **perfectly preserves norms**:
- Llama-3.1 -> Llama-3: max ||T_transferred|| - ||T_source|| = **0.0** (exact)
- Llama-3 -> Llama-3.1: max ||T_transferred|| - ||T_source|| = **7.63e-06** (float precision)

Source-norm rescaling is confirmed as a **no-op** (scale factors = 1.000000 for all tokens).

### Norm Statistics

| Token Set | Direction: 3.1->3 | Direction: 3->3.1 |
|-----------|-------------------|-------------------|
| Source DT | 85.35 +/- 3.28 | 83.02 +/- 2.17 |
| Procrustes-transferred | 85.35 +/- 3.28 | 83.02 +/- 2.17 |
| Target-native DT | 83.02 +/- 2.17 | 85.35 +/- 3.28 |

All DefensiveTokens have L2 norms in the ~80-91 range, which is approximately 100x the typical vocabulary embedding norm (~0.8-1.0). The high-norm property is preserved after transfer.

### ASR / RefusalRate Comparison

| Condition | Direction | Token L2 Norms (mean +/- std) | ASR (%) | RefusalRate (%) |
|-----------|-----------|-------------------------------|---------|-----------------|
| Procrustes (no rescaling) | 3.1 -> 3 | 85.35 +/- 3.28 | 0.0 | 0.5 |
| Source-norm rescaling | 3.1 -> 3 | 85.35 +/- 3.28 | 0.0 | 0.5 |
| Target-native-norm rescaling | 3.1 -> 3 | 83.02 +/- 2.17 | 0.0 | 0.5 |
| Procrustes (no rescaling) | 3 -> 3.1 | 83.02 +/- 2.17 | 33.7 | 1.0 |
| Source-norm rescaling | 3 -> 3.1 | 83.02 +/- 2.17 | 33.7 | 1.0 |
| Target-native-norm rescaling | 3 -> 3.1 | 85.35 +/- 3.28 | 34.1 | 1.0 |

## Key Observations

1. **Orthogonal Procrustes perfectly preserves norms**: Source-norm rescaling is numerically a no-op, confirming the theoretical property.

2. **Norm rescaling has no meaningful effect on defense quality**: All three conditions produce virtually identical ASR in both directions. The 0.4pp increase for target-native-norm rescaling in the 3->3.1 direction is within noise.

3. **The high-norm property (~100x vocabulary norm) is preserved and functional after transfer**: Both source and target DefensiveTokens share similar norm magnitudes (83-85), so the relative scale between embedding spaces is nearly identical.

4. **The 3->3.1 direction failure is not a norm problem**: The 33.7% ASR in the reverse direction is caused by directional/semantic mismatch, not by incorrect norm magnitudes. Rescaling norms to match the target-native tokens does not help.

5. **Conclusion**: For Procrustes-based transfer between closely related models with similar embedding scales, norm rescaling is unnecessary. The defense effectiveness depends on token direction (semantic content), not absolute norm magnitude, as long as the norms remain in the correct high-norm regime.
