
import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
import transformers

from rpc import enable_rpc, set_rpc_config
from utils.apply_chat_template import apply_chat_template

# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"


def gen_example(model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            rpc: bool = True,
            max_new_tokens: int = 32768,
            # RPC arguments
            P=1024,
            R=32,
            c=4,
            selectors='recent',
            aggregation='all',
            ):

    attn_implementation = 'flash_attention_2'
    if 'qwq' in model_path.lower():
        top_k = 40
    else:
        top_k = None

    print(f"Using Model: {model_path}, therefore top_k={top_k}")
    
    if rpc:
        enable_rpc()
    
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer)

    if rpc:
        set_rpc_config(model=model,
                            P=P,
                            R=32,
                            c=4,
                            selectors=selectors,
                            aggregation=aggregation,
                            kernel_size=7,
                            pooling='avgpool',
                            )
    else:
        print(["Full KV Cache Inference"])

    prompt = input("Prompt:")
    prompt = apply_chat_template(tokenizer, prompt)

    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                                do_sample=True,
                                temperature=0.6,
                                top_p=0.95,
                                top_k=top_k,
                                streamer=streamer)

    output_length = outputs[0][context_length:].shape[-1]
    print(f"\nContext Length: {context_length}")
    print(f"Output Length: {output_length}\n")


if __name__ == '__main__':
    
    fire.Fire(gen_example)    
