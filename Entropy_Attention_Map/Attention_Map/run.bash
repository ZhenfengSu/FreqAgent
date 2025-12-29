# https://zhuanlan.zhihu.com/p/1897638897388873200
MODEL_PATH="/mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B"
# # 运行脚本生成注意力图
# python block_attention.py

# # 从JSON文件加载第一条消息
# python block_attention.py --json trajectories.json

# # 从JSON文件加载指定索引的消息
# python block_attention.py --json trajectories.json --sample_idx 5

# 完整示例
python block_attention.py \
    --model /mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B \
    --json trajectories.json \
    --sample_idx 0 \
    --layers 20,25 \
    --output attention_map.png
