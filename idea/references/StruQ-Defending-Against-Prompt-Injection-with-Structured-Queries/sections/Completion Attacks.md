Completion Attacks
A strong attack is to first append a fake response to the prompt, misleading the LLM that the application's task has been completed, then inject new instructions, which the LLM tends to follow ~\cite{b16,b29}. We also insert appropriate delimiters to match the format of legitimate queries. We show an illustrative example: Completion-Real attack ### instruction: Is this email trying to sell me something? Answer yes or no.

Section references:
[b16]: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi. Tree of attacks: Jailbreaking black-box LLMs automatically. (2023). Tree of attacks: Jailbreaking black-box LLMs automatically
[b29]: Simon Willison. Delimiters won't save you from prompt injection. (2023). Delimiters won't save you from prompt injection