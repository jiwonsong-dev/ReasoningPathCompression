# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

for bsz in 8 16 32
do
    for ol in 4096 8192 16384 32768
    do  
        python -m benchmark.throughput \
        --model_path $MODEL \
        --batch_size $bsz \
        --output_len $ol

        python -m benchmark.throughput \
        --model_path $MODEL \
        --rpc True \
        --P 1024 \
        --R 32 \
        --c 4 \
        --batch_size $bsz \
        --output_len $ol
    done
done

