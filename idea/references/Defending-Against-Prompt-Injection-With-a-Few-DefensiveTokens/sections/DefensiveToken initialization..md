DefensiveToken initialization.
We also experiment with different initializations of the tuned tokens in Table6. It turns out that random initialization is better than the other heuristics like initializing with the embeddings of space and text (“You should follow all the instructions in the system block and not follow any instructions in the user block.” following ~\cite{bib.bib42}). Based on Table2 we hypothesize that it is because random initialization gives larger magnitude embeddings that facilitate optimization. If starting on a small initialization using vocabulary embeddings the optimizer needs to first enlarge those embeddings for a larger optimization space where a good solution lies. This conclusion on initialization is different from the original prompt tuning paper ~\cite{bib.bib15} where initializing with text embeddings works best. This may be because our defense objective is more complex than improving utility in a given task see Section3.4[ref_id]S3.SS4 and thus requires a larger optimization space.
[TABLE START]<table>
	<tr>
		<td>Embeddings in</td>
		<td>Avg 1-norm</td>
		<td>Max 1-norm</td>
	</tr>
	<tr>
		<td>Vocabulary Tokens</td>
		<td>34</td>
		<td>47</td>
	</tr>
	<tr>
		<td>Defensive Tokens</td>
		<td>4332</td>
		<td>4594</td>
	</tr>
</table>
Table 2. The magnitude of 4096-d embeddings in the Llama-3.1-8B-Instruct vocabulary vs. those in DefensiveToken.[TABLE END]

[TABLE START]<table>
	<tr>
		<td>Init.</td>
		<td>#Tokens</td>
		<td>WinRate (\uparrow)</td>
		<td>ASR (\downarrow)</td>
	</tr>
	<tr>
		<td>None</td>
		<td>0</td>
		<td>29.07</td>
		<td>69.23</td>
	</tr>
	<tr>
		<td>random</td>
		<td>1</td>
		<td>28.44</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>space</td>
		<td>1</td>
		<td>27.49</td>
		<td>7.7</td>
	</tr>
	<tr>
		<td>random</td>
		<td>5</td>
		<td>28.53</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>space</td>
		<td>5</td>
		<td>27.04</td>
		<td>2.40</td>
	</tr>
	<tr>
		<td>random</td>
		<td>20</td>
		<td>29.00</td>
		<td>0</td>
	</tr>
	<tr>
		<td>space</td>
		<td>20</td>
		<td>25.88</td>
		<td>0</td>
	</tr>
	<tr>
		<td>text</td>
		<td>20</td>
		<td>25.74</td>
		<td>0</td>
	</tr>
</table>
Table 6. Ablation study on the initialization of defensive tokens in DefensiveToken using AlpacaFarm and Llama3.1-8B-Instruct.[TABLE END]



## Section References
[bib.bib42] Wu et al. (2025a) Tong Wu Chong Xiang Jiachen T. Wang and Prateek Mittal. 2025a. Effectively Controlling Reasoning Models through Thinking Intervention. arXiv:2503.24370 https://arxiv.org/abs/2503.24370
[bib.bib15] Lester et al. (2021) Brian Lester Rami Al-Rfou and Noah Constant. 2021. The Power of Scale for Parameter-Efficient Prompt Tuning. In Empirical Methods in Natural Language Processing (EMNLP) Marie-Francine Moens Xuanjing Huang Lucia Specia and Scott Wen-tau Yih (Eds.). 3045–3059. doi:10.18653/v1/2021.emnlp-main.243 [https://doi.org/10.18653/v1/2021.emnlp-main.243]