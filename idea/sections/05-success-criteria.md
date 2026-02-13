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