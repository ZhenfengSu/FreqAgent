# 使用示例数据测试
# python frequency_analysis.py --use_sample


MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# 使用自定义数据
python frequency_analysis.py --data_path trajectories.json --model_name $MODEL_PATH

# 自定义层和目标长度
# python frequency_analysis.py --data_path trajectories.json --layers 20,22,24,26,28 --target_length 256