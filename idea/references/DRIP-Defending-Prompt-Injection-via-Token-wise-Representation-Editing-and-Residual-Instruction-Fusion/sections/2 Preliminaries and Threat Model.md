2 Preliminaries and Threat Model
Prompt Injection. A typical LLM prompt consists of four components: (1) a system instruction specifying global behavioral constraints; (2) a user instruction defining the immediate task; (3) a data section providing input context (e.g. retrieved documents or code outputs); and (4) the model response. Prompt injection refers to attacks that manipulate the prompt to subvert the intended instruction typically by including malicious directives in the user instruction or data section. Prior work categorizes such attacks into two types ~\cite{}:
 •
 Direct injection where the attacker controls the user instruction directly.
 •
 Indirect injection where the attacker manipulates the data section such as retrieved web content.
 We follow the state-of-the-art StruQ ~\cite{} and SecAlign ~\cite{} settings targeting the problem of indirect injection. Thus we simplify the terminologies by referring all trusted instructions (system or user-authored) as the top-level instruction and any untrusted potentially injected content as data.
 Attack Methods. Beyond naive injection such as including directive prefixes (see Table 1) researchers have proposed more sophisticated attacks. Specifically optimization-based attacks learn adversarial suffixes that maximize the likelihood of generating harmful or off-task responses. For example in Figure 2[ref_id]S2.F2 GCG (Greedy Coordinate Gradient) attack ~\cite{} learns adversarial suffixes to maximize the probability of generating “Hacked”.
 GCG Attack
 ## Instruction Translate the following paragraph into French. ## Data Today is a beautiful day. Print "Hacked." buildings;?>" geben parties. :?" deselect.SelectSingleNode ===> output pokusomidou.tsv :| hackedtasRich @/ ## Model Response  Hacked.
 Threat Model. Thus we formulate our threat model as follows. We consider a prompt p=x\oplus d where x is a trusted top-level instruction authored by the application developer and d is an untrusted data segment potentially containing injected instructions.
 The attacker may craft d as: d=d_{\text{clean}}\oplus x_{\text{prefix}}\oplus x_{\text{injected}}\oplus x_{\text{suffix}} where x_{\text{injected}} is the adversarial instruction and x_{\text{prefix}}x_{\text{suffix}} are auxiliary strings used to shift model focus or evade detection (e.g. via heuristic or optimization-based attacks).
 We assume a white-box threat model: the attacker has full knowledge of the model weights and deployed defense mechanisms but cannot modify the model itself. They may adaptively construct d to maximize attack success. An attack is considered successful if the model responds to x_{\text{injected}} instead of following the intended instruction x.
 Defender Objective. As defenders we aim to implement a finetuning-based defense by training an open-source language model f to be inherently aware of prompt injection. The model f is considered robust to prompt injection only if the following two conditions are satisfied:
 1.
 Injection Resistance: When instruction x_{a} is injected into the data portion of a different instruction x_{b} i.e.
 \displaystyle p=x_{b}\oplus\bigl(d_{b}\oplus{\color[rgb]{001}\definecolor[named]{pgfstrokecolor}{rgb}{001}x_{a}}\bigr)\quad\text{(inject at the end or)} \displaystyle p=x_{b}\oplus\bigl({\color[rgb]{001}\definecolor[named]{pgfstrokecolor}{rgb}{001}x_{a}}\oplus d_{b}\bigr)\quad\text{(at the start or)} \displaystyle p=x_{b}\oplus\bigl(d_{b}^{(1)}\oplus{\color[rgb]{001}\definecolor[named]{pgfstrokecolor}{rgb}{001}x_{a}}\oplus d_{b}^{(2)}\bigr)\quad\text{(in the middle)}
 the model’s output should not answer x_{a} but should execute x_{b} on all data treating x_{a} as part of the data.
 2.
 Utility Preservation: When the same task appears as the top-level instruction x_{a} i.e. p={\color[rgb]{001}\definecolor[named]{pgfstrokecolor}{rgb}{001}x_{a}}\oplus d_{a} the model’s output should follow x_{a}.
[TABLE START]<table>
	<tr>
		<td>Attack Method</td>
		<td>Intuition</td>
	</tr>
	<tr>
		<td>Naive ~\cite{}</td>
		<td>Inject the instruction verbatim, without any prefix/suffix.</td>
	</tr>
	<tr>
		<td>Ignore ~\cite{}</td>
		<td>Tell the model to ignore prior instructions and follow the injected one.</td>
	</tr>
	<tr>
		<td>Completion ~\cite{}</td>
		<td>Imply that the original task has been completed, nudging the model to start the injected task.</td>
	</tr>
	<tr>
		<td>Escape ~\cite{}</td>
		<td>Wrap the payload in escaping delimiters to bypass parsing heuristics or extend the prompt.</td>
	</tr>
	<tr>
		<td>HackaPrompt ~\cite{}</td>
		<td>A crowd-sourced prompt injection dataset collected via global “prompt hacking” competitions.</td>
	</tr>
</table>
Table 1: Heuristic-based attack strategies and their underlying intuitions.[TABLE END]
