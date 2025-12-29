# https://zhuanlan.zhihu.com/p/1897638897388873200
MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 只分析 Baseline
python block_attention.py --model_path $MODEL_PATH --input_json trajectories.json --output_dir ./output --sample_idx 0