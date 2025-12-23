MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 只分析 Baseline

python attention_analysis.py \
    --model_path $MODEL_PATH \
    --data_path trajectories.json  \
    --output_dir ./results \
    --max_samples 100


# python attention_analysis.py \
#     --model_path $MODEL_PATH \
#     --ablation_path $MODEL_PATH \
#     --data_path trajectories.json \
#     --output_dir ./comparison_results \
#     --max_samples 100 \
#     --compare