import os
import json
from datetime import datetime
import collections

from utils.apply_chat_template import apply_chat_template

def count_completed_samples(output_file):
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item['prompt']
                    gen_count = len(item.get('gen', []))
                    prompt_counts[prompt] += gen_count
                except json.JSONDecodeError:
                    continue
    return prompt_counts

def batched_generate(
    model,
    tokenizer,
    output_file: str,
    batch_dicts: list,
    batch_size: int = 4,
    max_new_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 40,
):   

    template_prompts = [apply_chat_template(tokenizer, d['prompt']) for d in batch_dicts]
    inputs = tokenizer(template_prompts, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]

    outs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )

    responses = [tokenizer.decode(o[input_length:], skip_special_tokens=True) for o in outs]

    for i, response in enumerate(responses):

        result = {}

        for key in batch_dicts[i].keys():
            result[key] = batch_dicts[i][key]
            result["gen"] = [response]

        with open(output_file, 'a', encoding='utf-8') as g:
            g.write(json.dumps(result, ensure_ascii=False) + '\n')
            g.flush()
