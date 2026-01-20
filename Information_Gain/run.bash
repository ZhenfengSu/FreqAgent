# 基本用法
python ig_calculator.py -i your_data.jsonl -o ig_results.json

# 使用 LLM judge（更准确但更慢）
python ig_calculator.py -i your_data.jsonl -o ig_results.json --use-llm-judge

# 减少采样次数（更快但估计可能不够稳定）
python ig_calculator.py -i your_data.jsonl -o ig_results.json -k 10

# 测试模式：只处理前 5 个问题
python ig_calculator.py -i your_data.jsonl -o ig_results.json -n 5

# 保存完整上下文（文件会变大）
python ig_calculator.py -i your_data.jsonl -o ig_results.json --save-context