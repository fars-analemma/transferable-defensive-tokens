Takeaway: What preserves utility?
(1) Token-wise representation editing enables fine-grained control. Replacing our token-wise editing with a global role offset (Table 7 row 3) significantly harms utility. Global offsets suppress all data tokens uniformly ignoring the fact that only certain tokens (e.g. “ignore previous instruction”) are more semantically risky. Our editing layer selectively attenuates high-salience tokens while preserving benign context improving instruction fidelity. Figure 7 visualizes this selective behavior.
 (2) Summation fusion is more stable than concatenation. Using concatenation (Table 7 row 4) introduces additional projections that disrupt the decoder distribution degrading output quality. Summation in contrast preserves dimensionality and allows smooth blending between instruction and context. The theoretical proof comparing the utility between two fusion is present in Appendix C[ref_id]A3.
 This reinforces our design principle: robustness gains should come with precise and minimal architectural edits.
[TABLE START]<table>
	<tr>
		<td>Variant</td>
		<td>Data

(Train)</td>
		<td>Loss

(Train)</td>
		<td>Shift

Type</td>
		<td>Fusion

Type</td>
		<td>SEP (%)</td>
		<td>Utility (%)</td>
		<td>GCG ASR (%)</td>
	</tr>
	<tr>
		<td>No Case 2</td>
		<td>Curated</td>
		<td>SFT</td>
		<td>Linear</td>
		<td>Sum</td>
		<td>58.50 
\downarrow 22.4</td>
		<td>71.87 
\downarrow 12.02</td>
		<td>0.00 
\downarrow 1.06</td>
	</tr>
	<tr>
		<td>No Case 3</td>
		<td>Orig SEP</td>
		<td>DPO</td>
		<td>Linear</td>
		<td>Sum</td>
		<td>81.00 
\uparrow 0.1</td>
		<td>85.01 
\uparrow 1.12</td>
		<td>69.90 
\uparrow 68.84</td>
	</tr>
	<tr>
		<td>Embedding shift</td>
		<td>Curated</td>
		<td>DPO</td>
		<td>Embedding</td>
		<td>Sum</td>
		<td>90.10 
\uparrow 9.2</td>
		<td>76.70 
\downarrow 7.19</td>
		<td>0.00 
\downarrow 1.06</td>
	</tr>
	<tr>
		<td>Concat fusion</td>
		<td>Curated</td>
		<td>DPO</td>
		<td>Linear</td>
		<td>Concat</td>
		<td>75.70 
\downarrow 5.2</td>
		<td>70.14 
\downarrow 13.75</td>
		<td>0.00 
\downarrow 1.06</td>
	</tr>
	<tr>
		<td>No fusion</td>
		<td>Curated</td>
		<td>DPO</td>
		<td>Linear</td>
		<td>None</td>
		<td>84.90 
\uparrow 4.0</td>
		<td>83.02 
\downarrow 0.87</td>
		<td>62.80 
\uparrow 61.74</td>
	</tr>
	<tr>
		<td>Default</td>
		<td>Curated</td>
		<td>DPO</td>
		<td>Linear</td>
		<td>Sum</td>
		<td>80.9</td>
		<td>83.89</td>
		<td>1.06</td>
	</tr>
</table>
Table 7: Ablation results on LLaMA-8B, assessing the contribution of data curation and architectural components to injection defense.
Each variant modifies one design element of DRIP while keeping others fixed.
SEP (%) measures semantic role separation on the SEP benchmark;
Utility (%) measures instruction-following accuracy on AlpacaEval 2.0;
GCG ASR (%) reports attack success rate under suffix-based gradient attacks.
Green arrows indicate improvements over the default, and Red arrows indicate degradations.[TABLE END]

[IMAGE START] [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/dc3f886a5cb1f0bde7818e78ad6cd530.png]  [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/59b5c845dd84b7347fb78e87828fbec6.png]  [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/3db01a5b1b258133215c185b91a952db.png]  [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/2b935c93b1d0dae5b6821258884d2632.png] Figure 7: Token-wise visualization of de-instruction shift magnitudes over the data segment.
\langle|\text{start\_header\_id}|\rangle\ \text{user}\ \langle|\text{end\_header\_id}|\rangle marks the start of the data segment.
Tokens with the top-10 largest \ell_{2} shifts are highlighted in red; the injected instruction is boxed in black.
DRIP selectively applies stronger shifts to boundary tokens and attention-drifting phrases (e.g., “ignore”, “disregard”).[IMAGE END]
