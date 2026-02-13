## Impact Statement

If successful, this work would make DefensiveTokens substantially easier to maintain across frequent model updates: a provider could optimize DefensiveTokens for one checkpoint and reuse them on closely related checkpoints with negligible additional compute. If transfer fails, the negative result is still decision-changing: it would suggest that DefensiveTokens are strongly model-specific even within a model family, motivating more expressive transfer mechanisms (e.g., learned projectors) if amortization is required.

---