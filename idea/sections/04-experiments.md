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