## Proposed Approach

### Overview

Let the source model \(M_s\) have a token embedding matrix \(E_s \in \mathbb{R}^{|V|\times d}\) and the target model \(M_t\) have \(E_t \in \mathbb{R}^{|V|\times d}\), where \(V\) is a shared vocabulary with identical token-id indexing. Let the optimized DefensiveTokens for \(M_s\) be \(T_s \in \mathbb{R}^{k\times d}\) (with \(k=5\) in the DefensiveTokens paper).

We compute a linear alignment \(W\in\mathbb{R}^{d\times d}\) from \(E_s\) to \(E_t\) using only vocabulary embeddings, then transfer the defensive soft tokens by:
\[
T_t = T_s W.
\]
We then deploy \(T_t\) on \(M_t\) by prepending these \(k\) soft tokens to the input in the same way as DefensiveTokens.

### Method Details

**1) Build aligned embedding pairs.**  
Construct matrices \(X, Y \in \mathbb{R}^{n\times d}\) by taking corresponding rows from the two embedding tables:
- \(X_i = E_s[\text{token\_id}_i]\)
- \(Y_i = E_t[\text{token\_id}_i]\)

We will use either the full shared vocabulary or a high-frequency subset (to reduce the influence of rare tokens).

**2) Orthogonal Procrustes alignment (norm-preserving).**  
We solve the orthogonal Procrustes problem:
\[
W^* = \arg\min_{W^\top W = I} \lVert XW - Y \rVert_F,
\]
where \(\lVert\cdot\rVert_F\) is the Frobenius norm (square-root of the sum of squared matrix entries). This has a closed-form solution:
- Compute \(M = X^\top Y\)
- Compute the singular value decomposition (SVD; factorization into orthogonal matrices and singular values): \(M = U\Sigma V^\top\)
- Set \(W^* = U V^\top\)

We use an orthogonal map because it preserves vector norms and angles, which is relevant for DefensiveTokens since their large norms appear important for robustness (DefensiveTokens, Table 2).

**3) Transfer DefensiveTokens.**  
Compute \(T_t = T_s W^*\) and prepend \(T_t\) as soft tokens at inference time.

**4) Norm-handling ablation.**  
As an ablation, after mapping we rescale each transferred token to match the source token’s \(\ell_2\) norm. This tests whether small embedding-scale differences between checkpoints affect transfer.

**5) Optional “tiny adaptation” (ablation, not required).**  
If mapping-only transfer yields partial improvements, perform \(\le 200\) gradient steps updating only \(T_t\) (not model weights) using the DefensiveTokens training objective. This tests whether alignment provides a good initialization that reduces per-target optimization cost.

### Key Innovations

1. **Cross-model transfer for prompt injection defense:** Applies soft-prompt transfer to a security setting (prompt injection robustness) rather than task accuracy.
2. **Training-free alignment from embedding tables:** Uses a closed-form Procrustes alignment computed from vocabulary embeddings, without training a projector or using task labels.
3. **Decision-oriented evaluation:** Tests whether transferred tokens close a large fraction of the “no defense → full DefensiveTokens” security gap at near-zero per-checkpoint compute.

---