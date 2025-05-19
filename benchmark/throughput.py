import fire
import logging
from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpc import enable_rpc, set_rpc_config

from utils.qwen2_norepeat import qwen2_flashattention2_norepeat_forward

def monkeypatch():
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flashattention2_norepeat_forward

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        return sum(numbers) / len(numbers)
    
    numbers_excluding_min_max = numbers.copy()
    numbers_excluding_min_max.remove(min(numbers))
    numbers_excluding_min_max.remove(max(numbers))

    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)

def measure_throughput(
    model_path: str = "Qwen/QwQ-32B",
    rpc: bool = True,
    # RPC arguments
    P: int = 1024,
    R: int = 32,
    c: int = 4,
    selectors: str = 'recent',
    # experiment arguments
    batch_size: int = 16,
    input_len: int = 128,
    output_len: int = 32768,
    num_warmups: int = 1,
    num_runs: int = 3
    ):

    num_gpus = torch.cuda.device_count()

    attn_implementation = 'flash_attention_2'
    if rpc:
        enable_rpc()
        
    else:
        monkeypatch()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    if rpc:
        set_rpc_config(model=model,
                            P=P,
                            R=R,
                            c=c,
                            selectors=selectors,
                            aggregation='all',
                            kernel_size=7,
                            pooling='avgpool'                            
                            )

    # Input Sequence      
    input_id = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)
    attn_mask = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)
    context_length = input_id.shape[-1]

    if num_warmups > 0:
        for i in range(num_warmups):

            print(f"Warm Up Run #{i}")

            total_time = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():

                outputs = model(input_id, attn_mask, past_key_values=None)
                        
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

                for _ in tqdm(range(output_len - 1), ncols=100):
                    start.record()
                    outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
        del outputs
        del past_key_values
        del pred_token_idx

        cleanup_memory()
    
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device=i)


    results_list = []

    for i in range(num_runs):

        print(f"Test Run #{i}")

        total_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():

            outputs = model(input_id, attn_mask, past_key_values=None)
                    
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

            for _ in tqdm(range(output_len - 1), ncols=100):
                start.record()
                outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
        throughput = batch_size * (output_len - 1) / (total_time / 1000)

        results_list.append(throughput)

        del outputs
        del past_key_values
        del pred_token_idx

        cleanup_memory()

    avg_throughput = average_excluding_min_max(results_list)

    total_max_memory = 0
    for i in range(num_gpus):
        max_mem = torch.cuda.max_memory_allocated(device=i)
        total_max_memory += max_mem


    print(f"Model: {model_path}")
    print(f"Mode: {mode}")

    if rpc:
        print(f"P={P}")
        print(f"R={R}")
        print(f"c={c}" )

    print(f"Batch Size={batch_size}")
    print(f"Input Length={input_len}, Output Length={output_len}")
    print(f"Number of Warm Up Runs={num_warmups}, Number of Test Runs={num_runs}")
    print(f"Average Throughput (tokens/sec)={avg_throughput:.2f}")
    print(f"Peak GPU Memory: {total_max_memory / 1000**2 / 1000:.2f} GB\n")


if __name__ == "__main__":
    fire.Fire(measure_throughput)

