# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import transformers
import numpy as np
import json

with open('defensivetokens.json', 'r') as f: defensivetokens = json.load(f)
chat_templates = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",

    'meta-llama/Llama-3.1-8B-Instruct': 
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n""",

    'tiiuae/Falcon3-7B-Instruct': 
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {%- for message in messages %}\n
        {{- '<|' + message['role'] + '|>\\n' + message['content'] | trim + '\\n\\n' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|assistant|>\\n' }}\n{%- endif %}\n""",
    
    'Qwen/Qwen2.5-7B-Instruct': 
    """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {%- for message in messages %}\n
        {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n""",
}

for model_name, defensivetoken in defensivetokens.items():
    output_dir = model_name + '-%dDefensiveTokens' % len(defensivetoken)
    print('Processing', model_name, 'with', len(defensivetoken), 'defensive tokens to', output_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name) #
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name) #
    
    defensivetoken = torch.tensor(np.array(defensivetoken)).to(model.device) #
    additional_special_tokens = {}
    for i in range(len(defensivetoken)): additional_special_tokens[f'[DefensiveToken{i}]'] = defensivetoken[i:i+1, :]
    num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(additional_special_tokens.keys())})
    
    model.resize_token_embeddings(len(tokenizer)) #
    for i in range(len(defensivetoken)):  #
        model.get_input_embeddings().weight.data[-len(defensivetoken)+i] = additional_special_tokens[f'[DefensiveToken{i}]'] #

    model.save_pretrained(output_dir) #
    tokenizer.chat_template = chat_templates[model_name]
    tokenizer.save_pretrained(output_dir)