# 使用默认配置
# python kv_frequency_analysis.py --model your_model_path --data your_data.json

MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 指定参数
python kv_frequency_analysis.py \
    --model $MODEL_PATH \
    --data trajectories.json \
    --target_length 4096 \
    --max_samples 100

# 创建测试数据
# python kv_frequency_analysis.py --create_test_data