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

---

*Blueprint sections to be added: Story Arc, Figures, Tables, Paper Outline*
