# 分析文件夹下所有 JSONL 文件
python fair_entropy.py -i /path/to/jsonl_folder --limit 700

# 诊断文件夹
python fair_entropy.py --diagnose -i /path/to/jsonl_folder

# 单个 JSONL 文件
python fair_entropy.py -i data.jsonl --debug

# 原有的 JSON 文件模式仍然支持
python fair_entropy.py -i data.json