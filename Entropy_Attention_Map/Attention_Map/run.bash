MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 只分析 Baseline
python attention_map.py \
    --model_path $MODEL_PATH \
    --json_path trajectories.json \
    --output_dir ./attention_results \
    --max_samples 100 \
    --layers "8,16,24,31" \
    --max_length 32768