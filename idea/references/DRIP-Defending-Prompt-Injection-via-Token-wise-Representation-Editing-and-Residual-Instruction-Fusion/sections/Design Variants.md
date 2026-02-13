Design Variants
(A) Data curation strategy. We test three training configurations:
 •
 No Case 2 in Section 3.1[ref_id]A3.EGx2: Omit the contrast between correct and mistaken execution. This would replace the DPO objective with a standard supervised finetuning (SFT) objective.
 •
 No Case 3 in Section 3.1[ref_id]A3.EGx2: Omit examples where the same task appears as true instruction. This falls back to the original SEP training benchmark.
 •
 Full (default): Uses Cases 1 2 and 3 with DPO contrast.
 (B) Architectural components. We test:
 •
 No Instruction Fusion (Section 3.3[ref_id]S3.SS3): Conventional decoding without the residual path.
 •
 Summation Fusion (Section 3.3[ref_id]S3.SS3): This is the default fusion choice.
 •
 Concat Fusion (Section 3.3[ref_id]S3.SS3): Use concatenation-based fusion to replace the summation fusion.
 •
 Embedding-level Shift (Section 3.1[ref_id]S3.SS1): Replace token-wise representation editing with global role offset similar to ISE ~\cite{}.