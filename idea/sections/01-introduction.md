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