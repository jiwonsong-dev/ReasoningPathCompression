# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq
N_SAMPLES=1
BSZ=8

P=1024
R=32
c=4
SELECTORS=recent
AGGREGATION=all

python -m eval.generate_answers.infer_hf \
        --input_file "eval/data/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval_n1_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --P $P \
        --R $R \
        --c $c \
        --selectors $SELECTORS \
        --aggregation $AGGREGATION

P=4096
R=32
c=4
SELECTORS=recent
AGGREGATION=all

python -m eval.generate_answers.infer_hf \
        --input_file "eval/data/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval_n1_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --P $P \
        --R $R \
        --c $c \
        --selectors $SELECTORS \
        --aggregation $AGGREGATION