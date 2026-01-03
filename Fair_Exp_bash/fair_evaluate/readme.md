# 单个文件夹评测
```bash
# 基本用法
./run_evaluate.sh -i ./predictions/experiment1

# 指定数据集类型
./run_evaluate.sh -i ./predictions/experiment1 -d browsecomp_zh

# 完整参数
./run_evaluate.sh \
    -i ./predictions/experiment1 \
    -d gaia \
    -a http://127.0.0.1:6002/v1 \
    -m qwen2.5-72b-instruct \
    -w 20
```


# 多个文件夹批量评测
```bash
./batch_evaluate.sh gaia ./exp1 ./exp2 ./exp3
```


# 直接使用Python脚本
```bash
python evaluate.py \
    --input_folder ./predictions/experiment1 \
    --dataset gaia \
    --api_base http://127.0.0.1:6002/v1 \
    --model_name qwen2.5-72b-instruct \
    --max_workers 10
```

# 输出示例
评测完成后，results.json 的内容示例：
```json
{
  "dataset": "gaia",
  "input_folder": "./predictions/experiment1",
  "files": ["iter1.jsonl", "iter2.jsonl", "iter3.jsonl", "iter4.jsonl"],
  "statistics": {
    "total_questions": 103,
    "total_answers": 369,
    "num_rounds": 4,
    "avg_attempts_per_question": 3.58
  },
  "metrics": {
    "best_pass_at_1": 45.63,
    "best_pass_round": "round_2",
    "any_pass": 62.14,
    "any_pass_correct_count": 64
  },
  "per_round": {
    "round_1": {"correct": 42, "answered": 103, "total": 103, "pass_rate": 40.78},
    "round_2": {"correct": 47, "answered": 103, "total": 103, "pass_rate": 45.63},
    "round_3": {"correct": 44, "answered": 103, "total": 103, "pass_rate": 42.72},
    "round_4": {"correct": 25, "answered": 60, "total": 103, "pass_rate": 24.27}
  }
}
```