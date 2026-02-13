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