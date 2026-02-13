Position to insert DefensiveTokens.
DefensiveTokens at the start of the LLM (before the begin_of_sentence token) is far better than those optimized and placed at the end of the input (the idea of prefilling defense ~\cite{bib.bib42}) see Table8. We hypothesize that inserting them at the beginning allows them to attend to all following tokens offering more control of the output same as in traditional prompt tuning ~\cite{bib.bib15}.
 Learning rate turns out to affect security a lot but not the utility see Table9. We tune the learning rates exponentially. 0.01 is clearly too small to lend a reasonable security. 0.1 as we used is a good choice for security and utility. Increasing to 1 destabilize the training and may give lower or higher utility and security in an unpredictable manner.
[TABLE START]<table>
	<tr>
		<td>Pos. in Inp.</td>
		<td>#Tokens</td>
		<td>Utility (\uparrow)</td>
		<td>ASR (\downarrow)</td>
	</tr>
	<tr>
		<td></td>
		<td>0</td>
		<td>29.07</td>
		<td>69.23</td>
	</tr>
	<tr>
		<td>start</td>
		<td>1</td>
		<td>28.44</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>end</td>
		<td>1</td>
		<td>10.74</td>
		<td>0</td>
	</tr>
	<tr>
		<td>start</td>
		<td>5</td>
		<td>28.53</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>end</td>
		<td>5</td>
		<td>5.08</td>
		<td>0</td>
	</tr>
	<tr>
		<td>start</td>
		<td>20</td>
		<td>29.00</td>
		<td>0</td>
	</tr>
	<tr>
		<td>end</td>
		<td>20</td>
		<td>14.56</td>
		<td>0</td>
	</tr>
</table>
Table 8. Ablation study on the position of DefensiveTokens using AlpacaFarm and Llama3.1-8B-Instruct.[TABLE END]

[TABLE START]<table>
	<tr>
		<td>LR</td>
		<td>#Tokens</td>
		<td>Utility (\uparrow)</td>
		<td>ASR (\downarrow)</td>
	</tr>
	<tr>
		<td>None</td>
		<td>0</td>
		<td>29.07</td>
		<td>69.23</td>
	</tr>
	<tr>
		<td>0.01</td>
		<td>1</td>
		<td>29.10</td>
		<td>71.63</td>
	</tr>
	<tr>
		<td>0.1</td>
		<td>1</td>
		<td>28.44</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>1</td>
		<td>1</td>
		<td>28.18</td>
		<td>11.06</td>
	</tr>
	<tr>
		<td>0.01</td>
		<td>5</td>
		<td>29.23</td>
		<td>23.56</td>
	</tr>
	<tr>
		<td>0.1</td>
		<td>5</td>
		<td>28.53</td>
		<td>0.48</td>
	</tr>
	<tr>
		<td>1</td>
		<td>5</td>
		<td>27.21</td>
		<td>3.37</td>
	</tr>
	<tr>
		<td>0.01</td>
		<td>20</td>
		<td>28.72</td>
		<td>22.60</td>
	</tr>
	<tr>
		<td>0.1</td>
		<td>20</td>
		<td>28.79</td>
		<td>7.7</td>
	</tr>
	<tr>
		<td>1</td>
		<td>20</td>
		<td>29.00</td>
		<td>0</td>
	</tr>
</table>
Table 9. Ablation study on the learning rate of optimizing DefensiveTokens using AlpacaFarm and Llama3.1-8B-Instruct.[TABLE END]



## Section References
[bib.bib42] Wu et al. (2025a) Tong Wu Chong Xiang Jiachen T. Wang and Prateek Mittal. 2025a. Effectively Controlling Reasoning Models through Thinking Intervention. arXiv:2503.24370 https://arxiv.org/abs/2503.24370
[bib.bib15] Lester et al. (2021) Brian Lester Rami Al-Rfou and Noah Constant. 2021. The Power of Scale for Parameter-Efficient Prompt Tuning. In Empirical Methods in Natural Language Processing (EMNLP) Marie-Francine Moens Xuanjing Huang Lucia Specia and Scott Wen-tau Yih (Eds.). 3045–3059. doi:10.18653/v1/2021.emnlp-main.243 [https://doi.org/10.18653/v1/2021.emnlp-main.243]