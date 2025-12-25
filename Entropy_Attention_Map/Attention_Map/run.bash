MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 只分析 Baseline
python block_attention.py --mode real --model $MODEL_PATH --json trajectories.json