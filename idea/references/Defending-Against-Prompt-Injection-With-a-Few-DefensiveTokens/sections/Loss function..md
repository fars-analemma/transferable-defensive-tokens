Loss function.
SecAlign~\cite{bib.bib5} uses preference optimization instead of supervised fine-tuning in StruQ. Besides training the LLM to prefer the response to the user instruction SecAlign also penalizes the response to the injection. This is an objective harder than StruQ SFT and we find that a few new embeddings are insufficient to learn that. Table7 shows that DefensiveToken using the SecAlign loss hurts utility significantly while achieving perfect security as in ~\cite{bib.bib5}. Thus we adopt StruQ loss in our design.
[TABLE START]<table>
	<tr>
		<td>Loss</td>
		<td>Opt. Var.</td>
		<td>WinRate (\uparrow)</td>
		<td>ASR (\downarrow)</td>
	</tr>
	<tr>
		<td>None</td>
		<td>None</td>
		<td>29.07</td>
		<td>69.23</td>
	</tr>
	<tr>
		<td>StruQ</td>
		<td>1 token emb</td>
		<td>28.44</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>SecAlign</td>
		<td>1 token emb</td>
		<td>18.70</td>
		<td>0</td>
	</tr>
	<tr>
		<td>StruQ</td>
		<td>5 token embs</td>
		<td>28.53</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>SecAlign</td>
		<td>5 token embs</td>
		<td>26.83</td>
		<td>0</td>
	</tr>
	<tr>
		<td>StruQ</td>
		<td>20 token embs</td>
		<td>29.00</td>
		<td>0</td>
	</tr>
	<tr>
		<td>SecAlign</td>
		<td>20 token embs</td>
		<td>19.61</td>
		<td>0</td>
	</tr>
	<tr>
		<td>StruQ</td>
		<td>LoRA</td>
		<td>27.63</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>SecAlign</td>
		<td>LoRA</td>
		<td>27.47</td>
		<td>0</td>
	</tr>
	<tr>
		<td>StruQ</td>
		<td>Full</td>
		<td>28.24</td>
		<td>0</td>
	</tr>
</table>
Table 7. Ablation study on the loss in DefensiveToken using AlpacaFarm and Llama3.1-8B-Instruct.[TABLE END]



## Section References
[bib.bib5] Chen et al. (2025b) Sizhe Chen Arman Zharmagambetov Saeed Mahloujifar Kamalika Chaudhuri David Wagner and Chuan Guo. 2025b. SecAlign: Defending Against Prompt Injection with Preference Optimization. In The ACM Conference on Computer and Communications Security (CCS). https://arxiv.org/abs/2410.05451