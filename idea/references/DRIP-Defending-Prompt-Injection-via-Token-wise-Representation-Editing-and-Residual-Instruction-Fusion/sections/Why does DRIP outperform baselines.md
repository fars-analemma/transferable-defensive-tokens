Why does DRIP outperform baselines?


#### DRIP v.s. SecAlign.
DRIP v.s. SecAlign.
 Figure 14[ref_id]A3.F14 qualitatively compares DRIP to the strongest baseline SecAlign. Both use DPO-style contrastive supervision but SecAlign performs a global preference optimization updating all model parameters to suppress responses influenced by injected instructions. This often leads to over-generalized suppression: SecAlign under-generates even on clean prompts (benign instructions without injected suffixes) as shown in the example.
 In contrast DRIP localizes preference learning to the data section via a targeted representation-editing layer confining suppression to an embedding subspace while preserving the semantics of the instruction section by construction. Instruction fusion further reinforces the intended task at the logit level improving robustness against adaptive attacks where SecAlign remains vulnerable.

#### DRIP v.s. ISE.
DRIP v.s. ISE.
 ISE edits representations using a single global offset \{b} applied uniformly e_{\text{ISE}}(\{x})=e(\{x})+\{b}_{role}. To reliably “de-instruct” all instruction-like tokens this offset must be large enough to push even the most instruction-aligned embeddings across the boundary between \{M}_{instr} and \{M}_{data} i.e. it is determined by the worst-case token over the entire training distribution (see Appendix A.5[ref_id]A1.SS5). In practice minibatch training only sees local batches and thus tends to either underestimate the required shift or overshoot with an overly aggressive offset. Moreover because ISE applies the same offset to all roles many benign tokens are unnecessarily perturbed degrading utility.
 DRIP instead learns a token-wise correction g(e(\{x}_{a})). This allows strong edits only on instruction-like tokens while leaving neutral or descriptive tokens nearly unchanged yielding a cleaner separation between directive and data semantics. Figure 6 visualizes instruction-like tokens before and after editing: DRIP produces two linearly separable manifolds whereas under ISE they cannot be separated by a single hyperplane without errors.
[IMAGE START] [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/bfecd0b71b1a062acd388b809c9aa3d0.png]  [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/18af58eb6257bb857702c11a9dcc8b1d.png] Figure 6: T-SNE visualization of the representation editing for DRIP (Left),
and the role embedding offset by ISE (Right).[IMAGE END]
