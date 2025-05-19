import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import copy
import concurrent.futures
import threading
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpc import enable_rpc, set_rpc_config
from eval.generate_answers.utils_hf import count_completed_samples, batched_generate

def main():
    parser = argparse.ArgumentParser(description="Run inference on model with prompts from a jsonl file")
    parser.add_argument("--input_file", type=str, required=True, help="Input jsonl file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch per single call")
    parser.add_argument("--model_path", type=str, default='Qwen/QwQ-32B', help="Model name")
    parser.add_argument("--rpc", action="store_true", help="Run RPC")
    parser.add_argument("--P", type=int, default=512, help="Compression period")
    parser.add_argument("--R", type=int, default=128, help="Size of selector window size")
    parser.add_argument("--c", type=int, default=128, help="Target compression ratio")
    parser.add_argument("--selectors", type=str, default='recent', help="Selection policy")
    parser.add_argument("--aggregation", type=str, default='group', help="Aggregation policy")
    parser.add_argument("--kernel_size", type=int, default=7, help="Local pooling size")
    parser.add_argument("--pooling", type=str, default='avgpool', help="Type of local pooling")
    args = parser.parse_args()

    attn_implementation = 'flash_attention_2'
    if 'qwq' in args.model_path.lower():
        top_k = 40
    else:
        top_k = None

    print(f"Using Model: {args.model_path}, therefore top_k={top_k}")

    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file)
        total_completed = sum(completed_counts.values())
        print(f"Found {total_completed} completed samples from previous run")
    else:
        output_dir = Path(args.output_file).parent
        os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w', encoding='utf-8') as g:
            completed_counts = dict()
    
    # Load dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    expanded_data = []
    for item in data:
        prompt = item['prompt']
        completed = completed_counts.get(prompt, 0)
        remaining = max(args.n_samples - completed, 0)
        for _ in range(remaining):
            expanded_data.append(copy.deepcopy(item))
    
    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    
    if args.rpc:
        enable_rpc()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = 'left'

    if args.rpc: 
        set_rpc_config(model=model,
                            P=args.P,
                            R=args.R,
                            c=args.c,
                            selectors=args.selectors,
                            aggregation=args.aggregation,
                            kernel_size=args.kernel_size,
                            pooling=args.pooling,
                            )

    else:
        print(f"Full KV Cache Inference")

    for i in tqdm(range(0, len(expanded_data), args.batch_size)):

        batch_dicts = expanded_data[i : i + args.batch_size] 

        processing = len(batch_dicts)
        print(f"[Timestamp: {datetime.now()}][{total_tasks} samples remaining]")
        print(f"[Timestamp: {datetime.now()}][{processing} samples on processing]")
        
        batched_generate(
            model=model,
            tokenizer=tokenizer,
            output_file=args.output_file,
            batch_dicts=batch_dicts,
            batch_size=args.batch_size,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=top_k
        )

        total_tasks -= processing


if __name__ == "__main__":
    main()

    