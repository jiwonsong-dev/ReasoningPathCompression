
import os
import json
import fire
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
import transformers
from datasets import load_dataset
from collections import Counter

from utils.entropy import ngram_entropy
from utils.apply_chat_template import apply_chat_template

def get_entropy(model_path: str = "THUDM/LongWriter-llama3.1-8b",
                 output_dir:str = "outputs/entropy",
                 genlen:int = 1024,
            ):


    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    streamer = TextStreamer(tokenizer)

    dataset = load_dataset("json", data_files="eval/long-generation/heuristic_text_generation_8k.jsonl")
    prompts = dataset['train']['instruction']

    entropy_sum = 0
    entropy_2gram_sum = 0
    entropy_3gram_sum = 0

    results = []
  
    for i, prompt in enumerate(prompts):

        print(f"\nProb #{i}")
        
        prompt = f"[INST]{prompt}[/INST]"

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]

        with torch.no_grad():
            outputs = model.generate(input['input_ids'], attention_mask=input['attention_mask'], max_new_tokens=genlen, do_sample=False)
        
        output_vector = outputs[0][context_length:]
        output_length = output_vector.shape[-1]
        entropy = ngram_entropy(output_vector, 1)
        entropy_2gram = ngram_entropy(output_vector, 2)
        entropy_3gram = ngram_entropy(output_vector, 3)
        
        entropy_sum += entropy
        entropy_2gram_sum += entropy_2gram
        entropy_3gram_sum += entropy_3gram

        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")
        print(f"Entropy = {entropy}, 2-gram Entropy = {entropy_2gram}, 3-gram Enthropy = {entropy_3gram}")

        results.append({
                        'id': i,
                        'context_length': context_length,
                        'output_length': output_length,
                        'entropy': entropy,
                        'entropy_2gram': entropy_2gram,
                        'entropy_3gram': entropy_3gram
        })

    entropy_mean = entropy_sum/len(prompts)
    entropy_2gram_mean = entropy_2gram_sum/len(prompts)
    entropy_3gram_mean = entropy_3gram_sum/len(prompts)
    print(f"Mean Entorpy = {entropy_mean}, Mean 2-gram Entropy = {entropy_2gram_mean}, Mean 3-gram Entropy = {entropy_3gram_mean}")
    
    # dump as output file
    output_path = os.path.join(output_dir, 'hellobench', model_path.split('/')[-1] + '_genlen{genlen}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    
    fire.Fire(get_entropy)    
