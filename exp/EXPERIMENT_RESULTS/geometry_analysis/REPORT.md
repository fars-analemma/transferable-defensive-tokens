# Geometry Analysis of DefensiveToken Embeddings

## Overview

This report analyzes the geometric properties of source, Procrustes-transferred, and natively-optimized DefensiveToken (DT) embeddings relative to the vocabulary embedding space. The analysis covers both transfer directions: Llama-3.1-8B-Instruct -> Llama-3-8B-Instruct and the reverse.

## 1. Norm Statistics

### L2 Norms

| Token Set | Llama-3.1 -> Llama-3 | Llama-3 -> Llama-3.1 |
|---|---|---|
| Source DT (mean +/- std) | 85.35 +/- 3.28 | 83.02 +/- 2.17 |
| Transferred DT (mean +/- std) | 85.35 +/- 3.28 | 83.02 +/- 2.17 |
| Native Target DT (mean +/- std) | 83.02 +/- 2.17 | 85.35 +/- 3.28 |
| Source Vocab (mean) | 0.671 | 0.591 |
| Target Vocab (mean) | 0.591 | 0.671 |

### L1 Norms (matching Table 2 from DefensiveTokens paper)

| Token Set | Llama-3.1 -> Llama-3 | Llama-3 -> Llama-3.1 |
|---|---|---|
| Source DT L1 mean | 4331.9 | 4192.8 |
| Transferred DT L1 mean | 4330.6 | 4193.1 |
| Native Target DT L1 mean | 4192.8 | 4331.9 |
| Target Vocab L1 mean | 30.16 | 34.26 |

### Norm Ratios (DT / Vocab Mean)

| Metric | Llama-3.1 -> Llama-3 | Llama-3 -> Llama-3.1 |
|---|---|---|
| Source DT L2 ratio | 144.4x | 123.7x |
| Transferred DT L2 ratio | 144.4x | 123.7x |
| Native Target DT L2 ratio | 140.5x | 127.1x |
| Source DT L1 ratio | 143.6x | 122.4x |
| Transferred DT L1 ratio | 143.6x | 122.4x |
| Native Target DT L1 ratio | 139.0x | 126.4x |

**Finding (a): Transferred tokens remain extreme high-norm outliers.** The L2 norm ratio is ~124-144x the vocabulary mean, and the L1 norm ratio is ~122-144x. These match the paper's Table 2 values (vocab avg L1 ~34, DT avg L1 ~4332). Orthogonal Procrustes preserves norms exactly (max L2 difference between source and transferred: <5e-6), so the transferred tokens automatically remain in the high-norm outlier regime.

## 2. Cosine Similarity Analysis

### Pairwise Cosine Similarity (per-token diagonal)

| Comparison | Llama-3.1 -> Llama-3 (mean) | Llama-3 -> Llama-3.1 (mean) |
|---|---|---|
| Source vs Native (raw, different spaces) | 0.0970 | 0.0970 |
| Source(aligned) vs Native | 0.0968 | 0.0968 |
| Transferred vs Native | 0.0968 | 0.0968 |
| Source(aligned) vs Transferred | 1.0000 | 1.0000 |

### Per-Token Cosine Similarity (Transferred vs Native)

| Token | Llama-3.1 -> Llama-3 | Llama-3 -> Llama-3.1 |
|---|---|---|
| DT0 | 0.1104 | 0.1104 |
| DT1 | 0.0820 | 0.0820 |
| DT2 | 0.0963 | 0.0963 |
| DT3 | 0.1030 | 0.1030 |
| DT4 | 0.0923 | 0.0923 |

**Finding (b): Cosine similarity between transferred and native tokens is low (~0.097).** This means the transferred tokens point in substantially different directions compared to natively-optimized tokens. However, the defense still works (at least in the 3.1->3 direction with 0% ASR), suggesting that the defense mechanism relies more on the extreme norm magnitude than on the precise direction.

## 3. Procrustes Alignment Quality

| Metric | Llama-3.1 -> Llama-3 | Llama-3 -> Llama-3.1 |
|---|---|---|
| ||XW* - Y||_F | 71.48 | 71.48 |
| ||Y||_F | 213.31 | 242.30 |
| Relative residual (%) | 33.51% | 29.50% |

**Finding (c): The Procrustes alignment has a moderate relative residual (~30-34%).** This indicates that while the two embedding spaces share substantial structure (they use the same tokenizer and are closely related models), they are not perfectly isomorphic. The alignment is identical in both directions (same residual Frobenius norm) since the same token pairs are aligned.

### Does Procrustes improve alignment vs no alignment?

The raw cosine similarity (source vs native, no alignment) is 0.0970, while after Procrustes alignment it is 0.0968 -- essentially identical. This confirms what was found in the direct-copy ablation: the rotation difference between these two model variants is negligible. The Procrustes alignment provides almost no benefit in terms of directional alignment of the DefensiveTokens, because the embedding spaces are already nearly aligned rotationally.

## 4. Per-Token Transfer Analysis

### Cross-token cosine similarity (5x5 heatmap, Llama-3.1 -> Llama-3 direction)

|  | Native DT0 | Native DT1 | Native DT2 | Native DT3 | Native DT4 |
|---|---|---|---|---|---|
| Trans DT0 | **0.110** | 0.127 | 0.074 | 0.093 | 0.098 |
| Trans DT1 | 0.117 | **0.082** | 0.106 | 0.130 | 0.135 |
| Trans DT2 | 0.100 | 0.124 | **0.096** | 0.114 | 0.147 |
| Trans DT3 | 0.103 | 0.101 | 0.093 | **0.103** | 0.115 |
| Trans DT4 | 0.043 | 0.151 | 0.076 | 0.082 | **0.092** |

**Finding (d): No strong diagonal dominance in the cosine similarity matrix.** The diagonal values (corresponding transferred-to-native token pairs) are not consistently higher than off-diagonal values. For example, Trans DT4 has its highest similarity with Native DT1 (0.151) rather than Native DT4 (0.092). This confirms that the individual token identities are not preserved through transfer -- the tokens are not meaningfully "paired" between models.

However, all cosine similarities (both diagonal and off-diagonal) are low (~0.04 to 0.15), indicating that both transferred and native tokens occupy a distinct region of the embedding space that is approximately orthogonal to where native tokens reside. The defense mechanism appears to be primarily driven by the extreme norm, not by directional precision.

## 5. Key Conclusions

1. **High-norm outlier regime is preserved**: Procrustes transfer maintains L2 norms exactly (to machine precision), keeping transferred tokens at ~124-144x the vocabulary norm. This is the defining characteristic of effective DefensiveTokens.

2. **Directional similarity is low but defense still works (one direction)**: Despite cosine similarity of only ~0.097 between transferred and native tokens, the 3.1->3 transfer achieves 0% ASR. This suggests the defense effect is dominated by norm magnitude rather than directional specifics.

3. **Procrustes provides negligible directional benefit**: The raw and aligned cosine similarities are virtually identical (~0.097 vs ~0.097), consistent with the direct-copy ablation showing both methods achieve similar results.

4. **Asymmetric transfer explains ASR gap**: The 3->3.1 direction has 33.7% ASR despite identical norm preservation, while 3.1->3 has 0% ASR. Since norms and cosine similarities are symmetric, the asymmetry must arise from how each model's attention/processing layers respond to high-norm tokens -- a property not captured by embedding geometry alone.

5. **No token-level specialization in transfer**: The 5x5 cosine heatmap shows no diagonal dominance, meaning individual token identities are scrambled during transfer. The collective statistical properties (extreme norms in a particular subspace) matter more than per-token directions.

## Figures

- `results/figures/geometry_pca.png`: PCA projection of vocabulary (5k sample) and DefensiveTokens
- `results/figures/geometry_norm_histogram.png`: L2 norm distribution with DT markers
- `results/figures/geometry_cosine_heatmap.png`: 5x5 cosine similarity heatmap
