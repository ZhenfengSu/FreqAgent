MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 只分析 Baseline
python block_attention.py --mode real --model $MODEL_PATH --json trajectories.json

# # 方案1：使用顺序分析（默认）
# python block_attention.py --mode real --model /path/to/model

# # 方案2：使用4bit量化 + 顺序分析
# python block_attention.py --mode real --model /path/to/model --use-4bit

# # 方案3：使用采样方式（最省显存）
# python block_attention.py --mode real --model /path/to/model --use-sampling

# # 方案4：分块处理长序列
# python block_attention.py --mode real --model /path/to/model --use-chunked