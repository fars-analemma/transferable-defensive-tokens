# AlignDefTok: Training-Free Transfer of DefensiveTokens via Embedding-Space Alignment

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: Neural Information Processing Systems (NeurIPS), International Conference on Machine Learning (ICML), International Conference on Learning Representations (ICLR), Association for Computational Linguistics (ACL), Empirical Methods in Natural Language Processing (EMNLP), or similar top AI conferences

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly integrated into applications that ingest **untrusted text** (retrieved documents, emails, web pages) and may take actions (e.g., tool calls, application programming interface (API) requests). **Prompt injection** refers to attacks where an adversary places instructions inside this untrusted text to override the developer’s intended instruction. The Open Worldwide Application Security Project (OWASP) Top 10 for LLM Applications lists prompt injection as a top risk (LLM01: Prompt Injection).

A common mitigation is to enforce a separation between **trusted instructions** (system/developer prompts) and **untrusted data** (external content). Some defenses implement this separation through training-time methods that modify model weights, such as supervised fine-tuning (SFT), Direct Preference Optimization (DPO; preference-based fine-tuning), reinforcement learning (RL), or parameter-efficient adaptations like LoRA (Low-Rank Adaptation). These approaches can be effective, but they must often be repeated for each new model checkpoint or model variant.

**DefensiveTokens** (Chen et al., 2025) introduced a test-time defense that does not change the model weights. It prepends a small number (typically \(k=5\)) of learned **soft tokens**—continuous embedding vectors inserted into the prompt, as in prompt tuning (optimizing prompt embeddings while keeping model weights frozen)—to steer the model away from following injected instructions. On multiple prompt-injection benchmarks, DefensiveTokens substantially reduce **attack success rate (ASR; lower is safer)** while maintaining instruction-following utility.

### The Problem

DefensiveTokens must be optimized separately for each target model because the soft tokens live in the model’s embedding space and are learned by backpropagation through the full LLM. In practice, developers and model providers frequently need to defend multiple closely related checkpoints (e.g., base-model updates, instruction-tuned variants, organization-specific fine-tunes). Re-optimizing DefensiveTokens for every checkpoint reduces their usability as a reusable, inference-time defense.

Prior work on prompt tuning studies when learned continuous prompts transfer across tasks or models, often using (i) a trained projector between embedding spaces, or (ii) training-free alignment methods based on shared structure in embeddings. However, these works mostly target task performance, not security robustness. DefensiveTokens also have unusually large embedding norms—approximately two orders of magnitude larger than typical vocabulary embeddings (DefensiveTokens, Table 2)—so it is unclear whether standard transfer methods can extrapolate to these out-of-distribution vectors.

### Key Insight and Hypothesis

**Hypothesis:** For closely related LLMs that share an identical tokenizer and embedding dimensionality, the “defense-relevant” directions encoded by DefensiveTokens are approximately preserved across model variants up to a linear change of basis. Therefore, an **embedding-space alignment** estimated from the two models’ vocabulary embedding matrices can map DefensiveTokens from a source model to a target model, recovering most of the security benefit **without** per-target backpropagation.

**Operational definition of “closely related” (for this proposal):**
1. The two models use the same tokenizer (text-to-token-id mapping) with the same token-id → token-string mapping (so embedding rows correspond to the same discrete tokens).
2. The token embedding dimensionality matches (\(d_s=d_t\)), enabling a direct linear map.

**Why this might fail:** DefensiveTokens are high-norm outliers. A linear alignment fitted on normal vocabulary embeddings may not generalize to such out-of-distribution vectors. Transfer could yield negligible security improvement (ASR close to “no defense”) or induce excessive refusals (utility degradation).

---

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

## Related Work

### Field Overview

Prompt injection defenses aim to prevent untrusted text from overriding a model’s intended instruction. Existing approaches include:

- **Training-time robustness methods**, which modify model weights to make models better at separating instructions from data (e.g., SFT, DPO, RL, and representation editing). These can provide strong robustness but require retraining for each checkpoint.
- **Prompting and data-marking defenses**, which modify the prompt format to delimit or transform untrusted spans (e.g., “reminder” prompts or datamarking). These are easy to deploy but are often brittle to adaptive attacks.
- **Detection-based defenses**, which try to detect prompt injection attempts before answering or acting. These introduce false positives/negatives and often add runtime overhead.
- **System-level and architectural defenses**, which enforce capability control or data provenance (often targeting tool-using agents).
- **Soft-prompt defenses**, such as DefensiveTokens, which use continuous prompt vectors at inference time to steer behavior without weight updates.

Security is commonly measured by **attack success rate (ASR; lower is safer)** on benchmarks that contain an intended instruction plus an injected instruction inside untrusted text. Examples used in the DefensiveTokens paper include AlpacaFarm (208 instruction+data examples with a fixed injected instruction appended to the data), SEP (9.1k instruction–data–injection triples testing instruction/data separation), and TaskTracker (~31k prompt-injection examples targeting task drift).

Separately, prompt tuning and soft-prompt transfer research studies when learned continuous prompts can be reused across tasks or models. This proposal connects these areas by asking whether a security-oriented soft prompt (DefensiveTokens) can be amortized across closely related model checkpoints using a simple embedding-space alignment.

### Related Papers

- **[Defending Against Prompt Injection With a Few DefensiveTokens](./references/Defending-Against-Prompt-Injection-With-a-Few-DefensiveTokens/meta/meta_info.txt)**: Introduces DefensiveTokens; strong test-time defense but requires per-model optimization.
- **[StruQ: Defending Against Prompt Injection with Structured Queries](./references/StruQ-Defending-Against-Prompt-Injection-with-Structured-Queries/meta/meta_info.txt)**: Training-time defense using structured query formatting and reserved tokens.
- **[DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion](./references/DRIP-Defending-Prompt-Injection-via-Token-wise-Representation-Editing-and-Residual-Instruction-Fusion/meta/meta_info.txt)**: Training-time representation editing method for prompt injection robustness.
- **[Attention Tracker: Detecting Prompt Injection Attacks in LLMs](./references/Attention-Tracker-Detecting-Prompt-Injection-Attacks-in-LLMs/meta/meta_info.txt)**: Training-free prompt injection detection using attention-based signals.
- **[SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451)**: Uses DPO to train models resistant to prompt injection; requires per-checkpoint fine-tuning.
- **[Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks](https://arxiv.org/abs/2507.02735)**: Provides an open-weight secured model family trained with updated recipes.
- **[Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs (BIPIA)](https://arxiv.org/abs/2312.14197)**: Benchmark for indirect prompt injection and evaluation of prompting-based defenses.
- **[Defending Against Indirect Prompt Injection Attacks With Spotlighting](https://arxiv.org/abs/2403.14720)**: Uses datamarking/encoding of untrusted context to reduce prompt injection success.
- **[SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in RAG](https://arxiv.org/abs/2601.11199)**: Retrieval-augmented generation (RAG) defense that selectively redacts sensitive retrieved content.
- **[IntentGuard: Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis](https://arxiv.org/abs/2512.00966)**: Uses an intent analysis module to separate instructions from untrusted spans.
- **[CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution](https://arxiv.org/abs/2602.07918)**: Uses causal attribution to identify and suppress untrusted-context influence.
- **[Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)**: Architectural capability-control approach to prevent data-to-action prompt injection.
- **[The Instruction Hierarchy](https://arxiv.org/abs/2404.13208)**: Trains models to prioritize system/developer instructions over user or tool text.
- **[GCG: Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)**: Optimization-based attack method (Greedy Coordinate Gradient) used in prompt injection evaluations.
- **[Ignore Previous Prompt: Attack Techniques for LLMs](https://arxiv.org/abs/2211.09527)**: Early survey of prompt injection and related attack patterns.
- **[More than you’ve asked for: A Comprehensive Analysis of Prompt Injection](https://arxiv.org/abs/2302.12173)**: Systematizes indirect prompt injection through external content.
- **[Soft Begging: Modular and Efficient Shielding of LLMs against Prompt Injection](https://arxiv.org/abs/2407.03391)**: Uses soft prompts as modular defenses against prompt injection and jailbreak-style attacks.
- **[PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning](https://arxiv.org/abs/2406.04478)**: Uses adversarial soft-prompt tuning to mitigate backdoors; related to security-oriented soft prompts.
- **[Prompt Tuning](https://arxiv.org/abs/2104.08691)**: Foundational method for learning continuous prompts for frozen models.
- **[P-Tuning v2](https://arxiv.org/abs/2110.07602)**: Improves prompt tuning stability and performance via prefix-based methods.
- **[SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://arxiv.org/abs/2110.07904)**: Transfers soft prompts to reduce prompt-tuning cost on new tasks.
- **[On Transferability of Prompt Tuning for NLP](https://arxiv.org/abs/2111.06719)**: Studies cross-model prompt transfer using trained projectors with task supervision.
- **[Zero-Shot Continuous Prompt Transfer](https://arxiv.org/abs/2310.01691)**: Training-free cross-model transfer via relative-to-anchor encoding and search.
- **[Ultra-Low-Dimensional Prompt Tuning via Random Projection](https://arxiv.org/abs/2502.04501)**: Studies low-dimensional parameterizations for prompt tuning and transfer.
- **[Prompt Contrastive Transformation](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.22/132115/Prompt-Contrastive-Transformation-An-Enhanced)**: Improves prompt transfer via transformation and contrastive separation.
- **[PromptBridge](https://arxiv.org/abs/2512.01420)**: Transfers discrete instruction prompts to mitigate model drift across versions.
- **[TextGrad](https://doi.org/10.1038/s41586-025-08661-4)**: Uses LLM feedback to optimize discrete prompts; used as a baseline in DefensiveTokens.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Training-time robustness | Modify model weights (e.g., SFT/DPO/LoRA) so instructions override untrusted data | StruQ, SecAlign, Meta SecAlign, DRIP | AlpacaFarm, SEP, TaskTracker, InjecAgent (a tool-integrated agent prompt-injection benchmark) | Must retrain per checkpoint; risk of over-refusal or shortcut learning |
| Prompting / marking | Prompt templates or datamarking to delimit untrusted spans | Spotlighting, Reminder/Sandwich, BIPIA | Prompt injection ASR on indirect-PI benchmarks | Often brittle; adaptive attacks can bypass |
| Detection / monitoring | Detect prompt injection and refuse or filter inputs | Attention Tracker, IntentGuard | Detection accuracy + downstream ASR | False positives/negatives; runtime overhead |
| System / architecture | Capability control and isolation for tool use | Defeating Prompt Injections by Design, SD-RAG | Agentic benchmarks, privacy metrics | Engineering complexity; may reduce utility |
| Soft-prompt defenses | Continuous prompt vectors trained to reduce prompt injection | DefensiveTokens, Soft Begging | ASR + utility metrics | Often model-specific; transfer unclear |
| Soft prompt transfer | Reuse continuous prompts across models | SPoT, On Transferability of Prompt Tuning, Zero-Shot Continuous Prompt Transfer, Ultra-Low-Dimensional Prompt Tuning via Random Projection, Prompt Contrastive Transformation | Mostly task-performance benchmarks | Usually not evaluated for security prompts |

### Closest Prior Work

1. **DefensiveTokens**: Establishes the defense we aim to amortize; explicitly notes per-model optimization as a limitation.
2. **On Transferability of Prompt Tuning (Su et al., 2022)**: Shows cross-model transfer using trained projectors with task supervision; our approach removes projector training and targets security prompts.
3. **Zero-Shot Continuous Prompt Transfer (Wu et al., 2024)**: Training-free cross-model transfer via relative-space encoding and search; evaluated on task prompts rather than high-norm security prompts.
4. **Soft Begging (Ostermann et al., 2024)**: Uses soft prompts for security, but does not study cross-model amortization.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DefensiveTokens | Optimize \(k\) soft tokens per model for prompt injection defense | Requires per-model backpropagation | Transfer tokens via embedding-space alignment | If defense directions are preserved across variants, alignment should recover them cheaply |
| Prompt-Transferability (Su et al., 2022) | Train a projector for prompt transfer | Requires task supervision and training | Use vocabulary-based Procrustes alignment (training-free) | No projector training; immediate transfer to new checkpoints |
| Zero-Shot Continuous Prompt Transfer (Wu et al., 2024) | Training-free transfer via relative-space encoding + search | Designed for task semantics; search can be slow; not tested on high-norm prompts | Use closed-form linear alignment with norm preservation | Orthogonal alignment is simple and preserves geometry |
| Soft Begging | Uses soft prompts as defenses | Not amortized across checkpoints | Add explicit cross-model transfer mechanism | Enables reuse across model updates |

---

## Experiments

### Experimental Setup

**Tokenizer compatibility check (required pre-step).**  
Before alignment, verify that the source and target tokenizers are identical (same vocabulary size and same token-id → token-string mapping). If they differ, this proposal’s Procrustes-based transfer is out of scope (would require anchor-based or learned-projector transfer).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct | Source/target model for within-family transfer |
| Llama-3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct | Source/target model for within-family transfer |
| (Optional extension) Llama-3.1-70B-Instruct | 70B | https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct | Cross-size stress test (optional) |

**Training Data (only if DefensiveToken embeddings must be re-optimized):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Cleaned Alpaca | Instruction-following data used to optimize DefensiveTokens in the original paper | ~51k | https://github.com/gururise/AlpacaDataCleaned | See dataset repository for license |

**Other Resources:**
- DefensiveTokens code and scripts: https://github.com/Sizhe-Chen/DefensiveToken

**Resource Estimate**:

- **Compute budget**: ≤150 graphics processing unit (GPU)-hours (conservative upper bound)
  - **DefensiveTokens optimization (if needed to obtain \(T_s\))**: The DefensiveTokens paper reports using 4×NVIDIA A100-80GB GPUs with PyTorch FSDP (Fully Sharded Data Parallel) for ~1 hour per model (≈4 GPU-hours, i.e., 4 GPUs × 1 hour) to optimize \(k=5\) tokens for one epoch on Cleaned Alpaca (`./references/Defending-Against-Prompt-Injection-With-a-Few-DefensiveTokens/sections/Training..md`).
  - **Embedding alignment**: compute \(X^\top Y\) and an SVD of a \(d\times d\) matrix (e.g., \(4096\times4096\)); negligible compared to LLM training/inference.
  - **Inference for AlpacaFarm evaluation**: 208 prompts × a small number of defense variants; expected to be a few GPU-hours at most on an 8B model. Optional GCG-ASR evaluation is more expensive because it runs an adaptive gradient-based attack, so it can be deferred until after the optimization-free ASR results are known.
  - **Optional tiny-adapt**: ≤200 steps updating only \(T_t\); expected ≤10 GPU-hours.
  - **Optional 70B extension**: could require tens to hundreds of GPU-hours depending on batch sizes and whether full DefensiveTokens optimization is repeated; not required for initial verification.

- **GPU memory**: 8B models fit on 1×A100-80GB; 70B may require tensor parallelism (splitting model computation across multiple GPUs) or FSDP across multiple GPUs.
- **API usage**: None required.
- **Azure OpenAI content filter compliance**: This proposal avoids using OpenAI models (gpt-*, o1-*, etc.) for any prompts or judging. The core evaluation uses AlpacaFarm, where ASR is computed via a deterministic string-match rule and does not require a model-based judge (as in DefensiveTokens). Optional extensions that require an automated judge (e.g., SEP, TaskTracker) should use a non-OpenAI judge model to avoid Azure content filtering.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AlpacaFarm (prompt injection subset) | 208 instruction+data samples where an injected instruction (e.g., “Print exactly Hacked!”) is appended to the data to test whether the model follows the injection | ASR (attack success rate under optimization-free attacks; lower is safer). Optional: GCG-ASR (attack success rate under the Greedy Coordinate Gradient (GCG) adaptive attack; lower is safer) | test | https://github.com/tatsu-lab/alpaca_farm | DefensiveTokens repo scripts + paper’s string-match rule (`Security evaluation..md`) |
| (Optional extension) SEP | Instruction–data separation benchmark (9.1k samples) where each example includes an instruction, a data passage, and an injected instruction inside the data | ASR (attack success rate; lower is safer) | test | https://openreview.net/pdf?id=8EtSBX41mt | Requires an automated judge; use a non-OpenAI judge model (e.g., Llama-3.1-70B-Instruct) if run |
| (Optional extension) TaskTracker | Prompt-injection benchmark (~31k) designed to measure “task drift” under injected instructions in retrieved text | ASR (attack success rate; lower is safer) | test | (see DefensiveTokens repo) | Requires an automated judge; use a non-OpenAI judge model if run |

### Main Results

**Tiered outcome interpretation (for the main AlpacaFarm experiment):**
- **Strong success**: gap-closed ≥0.85 with no large increase in refusal rate → alignment-only transfer is likely usable without re-optimization.
- **Partial success**: gap-closed in [0.50, 0.85) → alignment provides a useful initialization; consider tiny-adapt.
- **Failure**: gap-closed <0.50 or large refusal-rate increase → DefensiveTokens do not transfer via simple linear alignment.

#### Comparability Rules (CRITICAL)

All methods below use the same AlpacaFarm prompts, the same injection variants, and the same ASR computation rule (“attack succeeds if the lowercase response begins with ‘hacked’”) as described in the DefensiveTokens paper (`./references/Defending-Against-Prompt-Injection-With-a-Few-DefensiveTokens/sections/Security evaluation..md`).

#### Results Table

Numbers below are copied from **Table 4** of the DefensiveTokens paper artifact: `./references/Defending-Against-Prompt-Injection-With-a-Few-DefensiveTokens/sections/4.2. DefensiveToken As SOTA Test-Time Defense.md` (SOTA = state of the art). Transfer results are **TBD** (to be verified).

| Method | Base Model | Benchmark | ASR ↓ (lower is safer) | GCG-ASR ↓ (lower is safer) | Utility proxy: RefusalRate ↓ (lower is better) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| No defense | Llama-3-8B-Instruct | AlpacaFarm | 51.4 | 94.7 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Published baseline |
| Reminder | Llama-3-8B-Instruct | AlpacaFarm | 34.6 | 96.6 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Prompting baseline |
| Sandwich | Llama-3-8B-Instruct | AlpacaFarm | 56.7 | 100.0 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Prompting baseline |
| Full DefensiveTokens | Llama-3-8B-Instruct | AlpacaFarm | 0.5 | 37.5 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Per-model optimized baseline |
| **Ours: Procrustes-transferred DefensiveTokens** | Llama-3-8B-Instruct | AlpacaFarm | **TBD** | **TBD** | **TBD** | - | Transfer from Llama-3.1-8B-Instruct |
| Ablation: Direct-copy DefensiveTokens | Llama-3-8B-Instruct | AlpacaFarm | **TBD** | **TBD** | **TBD** | - | Copy embeddings without alignment |
| No defense | Llama-3.1-8B-Instruct | AlpacaFarm | 69.2 | 96.2 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Published baseline |
| Reminder | Llama-3.1-8B-Instruct | AlpacaFarm | 29.8 | 97.1 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Prompting baseline |
| Sandwich | Llama-3.1-8B-Instruct | AlpacaFarm | 60.6 | 100.0 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Prompting baseline |
| Full DefensiveTokens | Llama-3.1-8B-Instruct | AlpacaFarm | 0.5 | 24.6 | **TBD (needs re-run)** | DefensiveTokens Table 4 | Per-model optimized baseline |
| **Ours: Procrustes-transferred DefensiveTokens** | Llama-3.1-8B-Instruct | AlpacaFarm | **TBD** | **TBD** | **TBD** | - | Transfer from Llama-3-8B-Instruct |
| Ablation: Direct-copy DefensiveTokens | Llama-3.1-8B-Instruct | AlpacaFarm | **TBD** | **TBD** | **TBD** | - | Copy embeddings without alignment |

**Definition of RefusalRate:** the fraction of benign prompts for which the model output matches a standard refusal pattern (e.g., “I can’t help with that”, “I’m not able to”, etc.). Concretely, we will use the same 208 AlpacaFarm items but remove the injected instruction from the data field, then measure how often the defended model produces a refusal. This is a coarse automated check to ensure the defense does not reduce ASR primarily by refusing to answer.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Direct copy vs Procrustes | Compare transfer without vs with alignment | If embedding spaces differ by a rotation, Procrustes should outperform direct copy |
| Norm rescaling | Rescale mapped tokens to match source \(\ell_2\) norms | If embedding-scale differences matter, rescaling improves ASR without increasing refusals |
| Tiny-adapt (≤200 steps) | Few gradient steps on target tokens only | If alignment provides a good initialization, tiny-adapt closes remaining gap at much lower cost than full optimization |

### Analysis (Optional)

- **Geometry diagnostics:** Compare cosine similarity (normalized dot product measuring the angle between vectors) and norm statistics of \(T_s\), \(T_t\), and vocabulary embeddings; test whether transferred tokens remain high-norm outliers as in the original method.
- **Boundary conditions:** Test whether transfer works for 8B↔8B but fails for 8B→70B, and report the failure regime.

---

## Success Criteria

**Criterion 1: Transfer closes most of the security gap (AlpacaFarm).**
- Hypothesis: Procrustes-transferred tokens achieve a large fraction of the security improvement of full DefensiveTokens on the target model.
- Validation: Compute the gap-closed ratio  
  \[
  r = \frac{\text{ASR}_{\text{no}} - \text{ASR}_{\text{xfer}}}{\text{ASR}_{\text{no}} - \text{ASR}_{\text{full}}}
  \]
  where \(\text{ASR}_{\text{no}}\) is the “no defense” ASR, \(\text{ASR}_{\text{full}}\) is the per-model-optimized DefensiveTokens ASR, and \(\text{ASR}_{\text{xfer}}\) is the transferred-token ASR. Interpret \(r\) using the tiered rubric in the Experiments section.

**Criterion 2: Utility does not collapse.**
- Hypothesis: Transfer does not reduce ASR primarily by forcing refusals.
- Validation: RefusalRate under transferred tokens remains close to the target model’s baseline RefusalRate and does not increase substantially relative to full DefensiveTokens.

**Criterion 3: Compute amortization is material.**
- Hypothesis: Transfer reduces per-checkpoint defense cost.
- Validation: Mapping-only transfer requires no backpropagation; if tiny-adapt is used, it should require far fewer optimization steps than full DefensiveTokens training (≥5× fewer) while retaining a large fraction of the ASR improvement.

---

## Impact Statement

If successful, this work would make DefensiveTokens substantially easier to maintain across frequent model updates: a provider could optimize DefensiveTokens for one checkpoint and reuse them on closely related checkpoints with negligible additional compute. If transfer fails, the negative result is still decision-changing: it would suggest that DefensiveTokens are strongly model-specific even within a model family, motivating more expressive transfer mechanisms (e.g., learned projectors) if amortization is required.

---

## References

- [Defending Against Prompt Injection With a Few DefensiveTokens](./references/Defending-Against-Prompt-Injection-With-a-Few-DefensiveTokens/meta/meta_info.txt) - Chen et al., 2025
- [StruQ: Defending Against Prompt Injection with Structured Queries](./references/StruQ-Defending-Against-Prompt-Injection-with-Structured-Queries/meta/meta_info.txt) - Chen et al., 2025
- [DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion](./references/DRIP-Defending-Prompt-Injection-via-Token-wise-Representation-Editing-and-Residual-Instruction-Fusion/meta/meta_info.txt) - 2025
- [Attention Tracker: Detecting Prompt Injection Attacks in LLMs](./references/Attention-Tracker-Detecting-Prompt-Injection-Attacks-in-LLMs/meta/meta_info.txt) - Hung et al., 2024
- [SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451) - Chen et al., 2024/2025
- [Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks](https://arxiv.org/abs/2507.02735) - Chen et al., 2025
- [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813) - Debenedetti et al., 2025
- [Defending Against Indirect Prompt Injection Attacks With Spotlighting](https://arxiv.org/abs/2403.14720) - 2024
- [SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in RAG](https://arxiv.org/abs/2601.11199) - 2026
- [IntentGuard: Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis](https://arxiv.org/abs/2512.00966) - 2025
- [CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution](https://arxiv.org/abs/2602.07918) - 2026
- [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208) - Wallace et al., 2024
- [GCG: Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) - Zou et al., 2023
- [Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs (BIPIA)](https://arxiv.org/abs/2312.14197) - Yi et al., 2023/2025
- [Soft Begging: Modular and Efficient Shielding of LLMs against Prompt Injection](https://arxiv.org/abs/2407.03391) - Ostermann et al., 2024
- [PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning](https://arxiv.org/abs/2406.04478) - 2024
- [Prompt Tuning](https://arxiv.org/abs/2104.08691) - Lester et al., 2021
- [P-Tuning v2](https://arxiv.org/abs/2110.07602) - Liu et al., 2021
- [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer](https://arxiv.org/abs/2110.07904) - Vu et al., 2022
- [On Transferability of Prompt Tuning for NLP](https://arxiv.org/abs/2111.06719) - Su et al., 2022
- [Zero-Shot Continuous Prompt Transfer](https://arxiv.org/abs/2310.01691) - Wu et al., 2024
- [Ultra-Low-Dimensional Prompt Tuning via Random Projection](https://arxiv.org/abs/2502.04501) - 2025
- [Prompt Contrastive Transformation](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.22/132115/Prompt-Contrastive-Transformation-An-Enhanced) - Zhao et al., 2025
- [PromptBridge](https://arxiv.org/abs/2512.01420) - Wang et al., 2025
- [TextGrad](https://doi.org/10.1038/s41586-025-08661-4) - Yuksekgonul et al., 2025
