Takeaway: What contributes to robustness?
(1) Case 2 is essential for semantic learning. Removing Case 2 (Table 7 row 1) drastically reduces SEP score as the model loses contrastive signals between correct and mistaken executions. Without it the model cannot reliably separate directive from non-directive semantics.
 (2) Case 3 prevents over-suppression. Dropping Case 3 (Table 7 row 2) weakens robustness under adaptive attacks indicating that the model learns shortcut features such as data source origins rather than learning the true role separation.
 (3) Instruction fusion defends against suffix overrides. Without the residual fusion path (Table 7 row 5) GCG ASR spikes confirming that fusing the top-level instruction at decoding time is key to resisting adversarial suffixes. The theoretical reason behind this phenomenon is present in Appendix B[ref_id]A2.
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
