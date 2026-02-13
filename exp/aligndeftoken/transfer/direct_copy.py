"""Direct Copy transfer: use source DefensiveToken embeddings on target model without any transformation.

This is the simplest baseline for cross-model token reuse (T_t = T_s).
Used in ablation study to isolate the contribution of Procrustes alignment.
"""

import json
import logging
import os

import numpy as np
import torch
import transformers

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_LLAMA3 = """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n"""


class DirectCopyTransfer:
    def __init__(
        self,
        defensivetokens_path: str,
        source_model_name: str,
        target_model_name: str,
    ):
        self.defensivetokens_path = defensivetokens_path
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        self.T_s = None

    def load_source_tokens(self) -> np.ndarray:
        with open(self.defensivetokens_path) as f:
            dt = json.load(f)
        self.T_s = np.array(dt[self.source_model_name], dtype=np.float32)
        logger.info(f"Loaded source DefensiveTokens: shape={self.T_s.shape} from {self.source_model_name}")
        return self.T_s

    def integrate_into_model(self, output_dir: str) -> str:
        if self.T_s is None:
            raise RuntimeError("Must call load_source_tokens() first")

        logger.info(f"Loading target model: {self.target_model_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.target_model_name, torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.target_model_name)

        model_dtype = model.get_input_embeddings().weight.dtype
        T_tensor = torch.tensor(self.T_s, dtype=torch.float32).to(model_dtype).to(model.device)
        num_tokens = T_tensor.shape[0]

        special_token_names = [f"[DefensiveToken{i}]" for i in range(num_tokens)]
        tokenizer.add_special_tokens({"additional_special_tokens": special_token_names})

        model.resize_token_embeddings(len(tokenizer))
        for i in range(num_tokens):
            model.get_input_embeddings().weight.data[-num_tokens + i] = T_tensor[i]

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.chat_template = CHAT_TEMPLATE_LLAMA3
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved direct-copy model to {output_dir}")

        del model
        return output_dir


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Direct Copy DefensiveToken transfer (no alignment)")
    parser.add_argument("--defensivetokens_path", type=str, required=True)
    parser.add_argument("--source_model", type=str, required=True)
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    transferer = DirectCopyTransfer(
        defensivetokens_path=args.defensivetokens_path,
        source_model_name=args.source_model,
        target_model_name=args.target_model,
    )
    transferer.load_source_tokens()
    transferer.integrate_into_model(args.output_dir)
    logger.info("Direct copy transfer complete.")


if __name__ == "__main__":
    main()
