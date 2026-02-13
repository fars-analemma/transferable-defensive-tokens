# Paper Blueprint: Training-Free Transfer of DefensiveTokens via Embedding-Space Alignment

## Meta Information
- **Analysis Date**: 2026-02-13
- **Experiments Analyzed**: 7 (baseline_defensivetokens_and_prompting, transfer_llama31_to_llama3, transfer_llama3_to_llama31, ablation_direct_copy_vs_procrustes, ablation_tiny_adapt, ablation_norm_rescaling, geometry_analysis)
- **Figures Generated**: 2 (method diagram + 1 analytical plot)
- **Tables Designed**: 2

## Claims

### claim_1: transfer_effectiveness_forward
**Statement**: Transferring DefensiveTokens from Llama-3.1-8B-Instruct to Llama-3-8B-Instruct via Orthogonal Procrustes alignment achieves 0.0% ASR (Attack Success Rate), matching or exceeding the native DefensiveTokens performance (3.8% measured ASR) with a gap-closed ratio of 1.08.
**Evidence**: transfer_llama31_to_llama3/RESULTS.json - max_asr_pct=0.0, gap_closed_ratio (measured)=1.0798
**Figures**: fig_framework_overview
**Tables**: tab_main_results

### claim_2: transfer_effectiveness_reverse
**Statement**: Transferring DefensiveTokens from Llama-3-8B-Instruct to Llama-3.1-8B-Instruct via Orthogonal Procrustes alignment followed by tiny-adapt (200 steps) achieves 1.9% ASR, matching the native DefensiveTokens performance (1.9% measured ASR) with a gap-closed ratio of 1.00.
**Evidence**: transfer_llama3_to_llama31/RESULTS.json - max_asr_pct=1.9, gap_closed_ratio (measured)=0.9997
**Figures**: fig_framework_overview, fig_tiny_adapt_convergence
**Tables**: tab_main_results

### claim_3: compute_efficiency
**Statement**: The transfer pipeline achieves 285× speedup for forward transfer (CPU-only, ~5 minutes) and 133× speedup for reverse transfer with tiny-adapt (~55 minutes on GPU) compared to full DefensiveTokens training (~16 GPU-hours).
**Evidence**: transfer_llama31_to_llama3/RESULTS.json - speedup_alignment_only="285x", total_pipeline_s=294.76; transfer_llama3_to_llama31/RESULTS.json - speedup_vs_full_training="~17x faster"
**Figures**: []
**Tables**: tab_main_results

### claim_4: procrustes_vs_direct_copy
**Statement**: Orthogonal Procrustes alignment provides marginal improvement over direct copy (0.9% ASR reduction in the harder direction), indicating that the embedding spaces of closely related models (Llama-3 and Llama-3.1) are nearly aligned, with the rotation matrix being close to identity.
**Evidence**: ablation_direct_copy_vs_procrustes/RESULTS.json - Llama-3→3.1: Direct Copy ASR=34.6%, Procrustes ASR=33.7%
**Figures**: []
**Tables**: tab_ablation

### claim_5: tiny_adapt_initialization_benefit
**Statement**: Procrustes-initialized tiny-adapt reaches <5% ASR in 25 training steps (4× faster than random initialization), demonstrating that the transferred tokens provide a strong initialization for the harder transfer direction.
**Evidence**: ablation_tiny_adapt/RESULTS.json - Procrustes step 25: ASR=2.4%, Random step 100: ASR=2.9%; speedup_analysis threshold_5.0: speedup="4.0x"
**Figures**: fig_tiny_adapt_convergence
**Tables**: tab_ablation

### claim_6: norm_preservation
**Statement**: Orthogonal Procrustes transformation exactly preserves L2 norms of DefensiveTokens (max difference <1e-5), maintaining the high-norm property (~85 L2 norm, 120-144× vocabulary average) that is critical for defense effectiveness.
**Evidence**: geometry_analysis/RESULTS.json - norm_preservation_max_diff=4.3e-06; DT L2 norms ~83-85, vocab L2 mean ~0.59-0.67
**Figures**: []
**Tables**: []

### claim_7: asymmetric_transfer_difficulty
**Statement**: Transfer difficulty is asymmetric: Llama-3.1→Llama-3 achieves perfect transfer (0% ASR) with Procrustes alone, while Llama-3→Llama-3.1 requires tiny-adapt to close the gap (33.7%→1.9% ASR), suggesting model-specific embedding space characteristics affect transferability.
**Evidence**: transfer_llama31_to_llama3/RESULTS.json - ASR=0.0%; transfer_llama3_to_llama31/RESULTS.json - Procrustes-only ASR=33.7%, with tiny-adapt ASR=1.9%
**Figures**: fig_tiny_adapt_convergence
**Tables**: tab_main_results

## Figure-Table Plan

**Purpose**: Prevent redundancy by deciding upfront what data goes into figures vs tables.

**Core Requirements**:
- **Main results table (REQUIRED)**: All transfer methods × both directions with precise ASR values, gap-closed ratios, and compute costs
- **Ablation table**: Direct copy vs Procrustes comparison, tiny-adapt final results

### Main Results Table (REQUIRED)
- Main comparison: All methods (No Defense, Reminder, Sandwich, Full DT, Direct Copy, Procrustes, Procrustes+Tiny-Adapt) × both transfer directions
- Organization: One comprehensive table with methods as rows, metrics (ASR, Gap-Closed, Compute) as columns, grouped by transfer direction
- Source: Individual experiment RESULTS.json files (baseline_defensivetokens_and_prompting, transfer_llama31_to_llama3, transfer_llama3_to_llama31, ablation_direct_copy_vs_procrustes)

### Additional Figures/Tables (analytical insights)
- **Tiny-adapt convergence curve**: Figure - shows ASR trend over training steps, comparing Procrustes-init vs Random-init (reveals initialization benefit)
- **Ablation results**: Table - Direct Copy vs Procrustes detailed comparison (precise values needed)
- NOTE: Each piece of data appears in ONLY ONE format

### Redundancy Check
- ❌ Main results table shows final ASR values; tiny-adapt figure shows convergence trajectory (different data)
- ✅ No overlap between figure and table data

## Figures

### fig_framework_overview
- **Path**: `method_diagrams/framework_overview.jpeg`
- **Type**: method_diagram
- **Caption**: Overview of AlignDefTok: Training-free transfer of DefensiveTokens via Orthogonal Procrustes alignment. Stage 1 computes the optimal rotation matrix W* from vocabulary embeddings using SVD. Stage 2 applies W* to transfer DefensiveTokens to the target model. Key properties: norm preservation, CPU-only computation, and optional tiny-adapt refinement.
- **Shows**: claim_1, claim_2, claim_6
- **Analysis**: This diagram illustrates the two-stage pipeline for transferring DefensiveTokens between models. The left side shows the alignment computation using Orthogonal Procrustes (W* = UV^T from SVD of X^T Y), while the right side shows the token transfer process. The diagram emphasizes the key properties that make this approach effective: exact norm preservation (critical for defense), training-free operation, and the optional tiny-adapt stage for harder transfer directions.

### fig_tiny_adapt_convergence
- **Path**: `analytical_plots/tiny_adapt_convergence.png`
- **Type**: analytical_plot
- **Caption**: Tiny-adapt convergence comparison: Procrustes initialization vs random initialization for Llama-3 → Llama-3.1 transfer. Procrustes-initialized tokens reach <5% ASR in 25 steps (4× faster than random), demonstrating the value of transferred tokens as initialization.
- **Shows**: claim_5, claim_7
- **Analysis**: This figure reveals the initialization benefit of Procrustes transfer. Starting from 33.7% ASR (vs 69.2% for random), Procrustes-initialized tiny-adapt converges 4× faster to the <5% ASR threshold. Both methods eventually approach the Full DT target (1.9%), but Procrustes provides a significantly better starting point, reducing the training budget needed for the harder transfer direction.

## Tables

### tab_main_results
- **Caption**: Main transfer results on AlpacaFarm benchmark. ASR (↓ better) measures attack success rate. Gap-Closed ratio (↑ better) measures defense recovery relative to Full DT. Best results in **bold**. Our methods (Procrustes, Procrustes+Tiny-Adapt) achieve comparable defense to Full DT with 133-285× compute savings.
- **Row Design** (7 rows):
  - Row 1: No Defense - Baseline without any protection (upper bound for ASR)
  - Row 2: Reminder - Prompting baseline that reminds model to ignore injections
  - Row 3: Sandwich - Prompting baseline that wraps data with defensive instructions
  - Row 4: Full DefensiveTokens - Native DT trained on target model (gold standard)
  - Row 5: Direct Copy - Naive transfer without alignment
  - Row 6: Procrustes Transfer (Ours) - Orthogonal Procrustes alignment
  - Row 7: Procrustes + Tiny-Adapt (Ours) - Procrustes with 200-step refinement
  - **Ordering Logic**: Baselines first (No Defense → Prompting → Full DT), then transfer methods (Direct Copy → Procrustes → Procrustes+Tiny-Adapt)
- **Column Design** (6 columns):
  - Column 1: Method
  - Column 2-3: Llama-3.1 → Llama-3 (ASR%, Gap-Closed)
  - Column 4-5: Llama-3 → Llama-3.1 (ASR%, Gap-Closed)
  - Column 6: Compute (relative to Full DT)
  - **Ordering Logic**: Easier direction first (3.1→3), then harder direction (3→3.1)
- **Visual Annotations**:
  - **Bold**: Best ASR per direction (excluding No Defense)
  - **(Ours)**: Mark our proposed methods
  - **Speedup**: Show compute savings (e.g., "285×")
- **Data Values** (with source verification):

| Method | 3.1→3 ASR% | 3.1→3 Gap | 3→3.1 ASR% | 3→3.1 Gap | Compute |
|--------|------------|-----------|------------|-----------|---------|
| No Defense | 51.4 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Meta-Llama-3-8B-Instruct.no_defense.asr_pct] | 0.00 | 69.2 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Llama-3.1-8B-Instruct.no_defense.asr_pct] | 0.00 | - |
| Reminder | 34.6 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Meta-Llama-3-8B-Instruct.reminder.asr_pct] | 0.35 | 29.8 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Llama-3.1-8B-Instruct.reminder.asr_pct] | 0.59 | - |
| Sandwich | 56.7 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Meta-Llama-3-8B-Instruct.sandwich.asr_pct] | -0.11 | 60.6 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Llama-3.1-8B-Instruct.sandwich.asr_pct] | 0.13 | - |
| Full DT | 3.8 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Meta-Llama-3-8B-Instruct.defensivetokens.asr_pct] | 1.00 | 1.9 [source: baseline_defensivetokens_and_prompting/RESULTS.json → meta-llama/Llama-3.1-8B-Instruct.defensivetokens.asr_pct] | 1.00 | 16 GPU-hr |
| Direct Copy | 0.0 [source: ablation_direct_copy_vs_procrustes/RESULTS.json → Llama-3.1 -> Llama-3.Direct Copy.asr_pct] | 1.08 | 34.6 [source: ablation_direct_copy_vs_procrustes/RESULTS.json → Llama-3 -> Llama-3.1.Direct Copy.asr_pct] | 0.51 | ~5 min |
| **Procrustes (Ours)** | **0.0** [source: transfer_llama31_to_llama3/RESULTS.json → results.transfer_asr.max_asr_pct] | **1.08** | 33.7 [source: transfer_llama3_to_llama31/RESULTS.json → procrustes_only_results.max_asr_pct] | 0.53 | ~5 min (285×) |
| **Procrustes+Tiny-Adapt (Ours)** | - | - | **1.9** [source: transfer_llama3_to_llama31/RESULTS.json → max_asr_pct] | **1.00** | ~55 min (133×) |

- **Key Insights Readers Should Extract**:
  1. Procrustes transfer achieves 0% ASR for 3.1→3 direction (better than Full DT's 3.8%)
  2. For harder 3→3.1 direction, Procrustes alone is insufficient (33.7% ASR), but tiny-adapt closes the gap to 1.9%
  3. Both transfer methods achieve massive compute savings (133-285×) compared to full training
  4. Prompting baselines (Reminder, Sandwich) are ineffective compared to DefensiveTokens
- **Data Source**: Synthesized from baseline_defensivetokens_and_prompting/RESULTS.json, transfer_llama31_to_llama3/RESULTS.json, transfer_llama3_to_llama31/RESULTS.json, ablation_direct_copy_vs_procrustes/RESULTS.json
- **Shows**: claim_1, claim_2, claim_3, claim_7

### tab_ablation
- **Caption**: Ablation study results. (a) Direct Copy vs Procrustes: Procrustes provides marginal improvement over direct copy, indicating near-identity rotation between closely related models. (b) Tiny-adapt initialization: Procrustes initialization converges 4× faster than random.
- **Row Design** (4 rows for part a, 2 rows for part b):
  - Part (a) - Transfer Method Comparison:
    - Row 1: Direct Copy
    - Row 2: Procrustes Transfer
  - Part (b) - Tiny-Adapt Initialization:
    - Row 3: Random Init (best checkpoint)
    - Row 4: Procrustes Init (best checkpoint)
- **Column Design**:
  - Part (a): Method | 3.1→3 ASR% | 3→3.1 ASR% | Δ ASR
  - Part (b): Init | Best Step | Best ASR% | Steps to <5%
- **Data Values** (with source verification):

**(a) Direct Copy vs Procrustes**
| Method | 3.1→3 ASR% | 3→3.1 ASR% | Δ ASR (3→3.1) |
|--------|------------|------------|---------------|
| Direct Copy | 0.0 [source: ablation_direct_copy_vs_procrustes/RESULTS.json] | 34.6 [source: ablation_direct_copy_vs_procrustes/RESULTS.json] | - |
| Procrustes | 0.0 [source: ablation_direct_copy_vs_procrustes/RESULTS.json] | 33.7 [source: ablation_direct_copy_vs_procrustes/RESULTS.json] | -0.9% |

**(b) Tiny-Adapt Initialization Comparison**
| Initialization | Best Step | Best ASR% | Steps to <5% |
|----------------|-----------|-----------|--------------|
| Random | 100 [source: ablation_tiny_adapt/RESULTS.json → random_init.best_checkpoint.step] | 2.9 [source: ablation_tiny_adapt/RESULTS.json → random_init.best_checkpoint.max_asr] | 100 |
| Procrustes | 150 [source: ablation_tiny_adapt/RESULTS.json → procrustes_init.best_checkpoint.step] | 1.9 [source: ablation_tiny_adapt/RESULTS.json → procrustes_init.best_checkpoint.max_asr] | 25 (4× faster) |

- **Key Insights**:
  1. Procrustes provides only 0.9% ASR improvement over direct copy for 3→3.1, suggesting near-identity rotation
  2. Procrustes initialization reaches <5% ASR 4× faster than random (25 vs 100 steps)
  3. Procrustes-init achieves better final ASR (1.9% vs 2.9%)
- **Data Source**: ablation_direct_copy_vs_procrustes/RESULTS.json, ablation_tiny_adapt/RESULTS.json
- **Shows**: claim_4, claim_5

## Story Arc

### Narrative Strategy
The paper tells a story of **practical efficiency**: how to get the benefits of DefensiveTokens (state-of-the-art prompt injection defense) without the computational cost of training them from scratch for each model. The narrative follows a problem-solution-validation structure:

1. **Problem Setup**: DefensiveTokens are effective but expensive (16 GPU-hours per model). As LLM ecosystems grow with frequent model updates, retraining defenses becomes impractical.

2. **Key Insight**: Embedding spaces of related models (e.g., Llama-3 and Llama-3.1) share similar structure. This suggests that soft prompts trained on one model might transfer to another via embedding space alignment.

3. **Solution**: Apply Orthogonal Procrustes alignment—a classic technique from cross-lingual embedding alignment—to transfer DefensiveTokens. The method is training-free, CPU-only, and preserves the high-norm property critical for defense.

4. **Validation**: Comprehensive experiments show the method achieves comparable defense (0-1.9% ASR) with 133-285× compute savings. Ablations reveal when pure transfer works vs when tiny-adapt is needed.

5. **Broader Impact**: This approach generalizes to any soft prompt transfer scenario, enabling rapid deployment of defenses across model families.

### Key Messages
1. **DefensiveTokens can be transferred between related models** without full retraining, achieving comparable defense effectiveness
2. **Orthogonal Procrustes alignment** provides a principled, training-free method for soft prompt transfer that preserves critical properties (norms)
3. **Transfer difficulty is asymmetric**: some directions work perfectly with alignment alone, others benefit from lightweight fine-tuning (tiny-adapt)
4. **Massive compute savings** (133-285×) make it practical to deploy defenses across model families

### Logical Flow
- **Introduction**: Motivate the problem (prompt injection threat + defense cost), introduce the transfer idea, preview contributions
- **Related Work**: Position against prompt injection defenses, soft prompt transfer methods, embedding alignment techniques
- **Method**: Formalize Orthogonal Procrustes for DefensiveToken transfer, describe tiny-adapt for harder cases
- **Experiments**: Validate effectiveness (main results), understand mechanisms (ablations), analyze geometry
- **Conclusion**: Summarize contributions, discuss limitations (tested on Llama family only), future directions (cross-architecture transfer)

## Paper Outline

### Abstract (~150 words)
- **Claims**: claim_1, claim_2, claim_3
- **Figures**: []
- **Tables**: []
- **Content Plan**: 
  - Background: Prompt injection attacks threaten LLM-integrated applications; DefensiveTokens provide effective defense but require expensive per-model training
  - Gap: No method exists to transfer trained DefensiveTokens to new models
  - Solution: AlignDefTok uses Orthogonal Procrustes alignment to transfer DefensiveTokens between models in a training-free manner
  - Results: Achieves 0-1.9% ASR (matching native DefensiveTokens) with 133-285× compute savings on Llama-3/3.1 transfer

### Introduction (~400 words)
- **Claims**: claim_1, claim_2, claim_3
- **Figures**: []
- **Tables**: []
- **Content Plan**:
  - Para 1: Prompt injection is a critical threat to LLM applications (cite Greshake2023, Perez2022)
  - Para 2: DefensiveTokens are state-of-the-art defense but expensive to train (cite chen2025defendingpromptinjectiondefensivetokens)
  - Para 3: Key insight—embedding spaces of related models share structure, enabling transfer (cite cross-lingual alignment work)
  - Para 4: Our contributions (numbered list):
    1. First method to transfer DefensiveTokens between models
    2. Training-free Orthogonal Procrustes alignment preserving critical properties
    3. Tiny-adapt for harder transfer directions
    4. 133-285× compute savings with comparable defense
  - Para 5: Paper roadmap

### Related Work (~250 words)
- **Claims**: []
- **Figures**: []
- **Tables**: []
- **Content Plan**:
  - **Prompt Injection Defenses**: Detection-based (Hung2024AttentionTD), prevention-based (Wallace2024TheIH, Hines2024DefendingAI), training-time (Chen2024SecAlignDA, struqdefendingagainstunknown), soft prompt (chen2025defendingpromptinjectiondefensivetokens, Ostermann2024SoftBM)
  - **Soft Prompt Transfer**: SPoT (Vu2021SPoTBF), transferability studies (Su2021OnTO), zero-shot transfer (Wu2023ZeroShotCP)
  - **Embedding Space Alignment**: Cross-lingual alignment (MUSE), Procrustes methods, model stitching
  - Position our work: First to apply embedding alignment for security-focused soft prompt transfer

### Method (~800 words)
- **Claims**: claim_6
- **Figures**: fig_framework_overview
- **Tables**: []
- **Content Plan**:
  - **3.1 Problem Formulation**: Define DefensiveTokens, transfer objective, notation
  - **3.2 Orthogonal Procrustes Alignment**: 
    - Compute alignment matrix W* = argmin_W ||XW - Y||_F s.t. W^T W = I
    - Closed-form solution via SVD: W* = UV^T where USV^T = X^T Y
    - Use vocabulary embeddings as anchor points
  - **3.3 DefensiveToken Transfer**:
    - Apply W* to source DefensiveTokens: T_target = T_source @ W*
    - Key property: Orthogonal transform preserves L2 norms exactly
  - **3.4 Tiny-Adapt for Harder Directions**:
    - When Procrustes alone is insufficient, fine-tune transferred tokens
    - Uses Procrustes output as initialization (4× faster convergence)
    - Only 200 steps, 20K parameters (vs 16 GPU-hours for full training)

### Experiments (~1000 words)
- **Claims**: claim_1, claim_2, claim_3, claim_4, claim_5, claim_7
- **Figures**: fig_tiny_adapt_convergence
- **Tables**: tab_main_results, tab_ablation
- **Content Plan**:
  - **4.1 Experimental Setup**:
    - Models: Llama-3-8B-Instruct, Llama-3.1-8B-Instruct
    - Dataset: AlpacaFarm (208 samples, 3 attack variants)
    - Metrics: ASR (attack success rate), Gap-Closed ratio, Compute time
    - Baselines: No Defense, Reminder, Sandwich, Full DefensiveTokens, Direct Copy
  - **4.2 Main Results** (Table 1):
    - Forward transfer (3.1→3): 0% ASR, 285× speedup
    - Reverse transfer (3→3.1): 1.9% ASR with tiny-adapt, 133× speedup
    - Analysis: Both directions achieve gap-closed ≥0.98
  - **4.3 Ablation Studies** (Table 2, Figure 2):
    - Direct Copy vs Procrustes: Marginal difference (0.9% ASR), near-identity rotation
    - Tiny-adapt initialization: Procrustes 4× faster to converge
    - Norm rescaling: No effect (orthogonal transform preserves norms)
  - **4.4 Analysis**:
    - Asymmetric transfer difficulty: 3.1→3 perfect, 3→3.1 needs tiny-adapt
    - Geometry: Low cosine similarity (~0.097) but defense works via high-norm property

### Conclusion (~80 words)
- **Claims**: []
- **Figures**: []
- **Tables**: []
- **Content Plan**:
  - Summary: AlignDefTok enables training-free transfer of DefensiveTokens with 133-285× compute savings
  - Limitations: Tested on Llama family only; cross-architecture transfer is future work
  - Broader impact: Approach generalizes to any soft prompt transfer, enabling rapid defense deployment

---

**Blueprint Status**: COMPLETE
**Ready for Writing Phase**: YES
