#!/bin/bash

# ============================================
# 批量评测脚本 - 评估多个实验文件夹
# ============================================

# 配置
API_BASE="http://127.0.0.1:6002/v1"
API_KEY="EMPTY"
MODEL="qwen2.5-72b-instruct"
WORKERS=10

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <dataset> <folder1> [folder2] [folder3] ..."
    echo ""
    echo "示例:"
    echo "  $0 gaia ./exp1 ./exp2 ./exp3"
    echo "  $0 browsecomp_zh ./results/model_a ./results/model_b"
    exit 1
fi

DATASET=$1
shift

echo "============================================"
echo "批量评测开始"
echo "数据集: $DATASET"
echo "待评测文件夹: $@"
echo "============================================"
echo ""

# 遍历所有文件夹
for FOLDER in "$@"; do
    if [ -d "$FOLDER" ]; then
        echo ">>> 正在评测: $FOLDER"
        ./run_evaluate.sh -i "$FOLDER" -d "$DATASET" -a "$API_BASE" -m "$MODEL" -w "$WORKERS"
        echo ""
    else
        echo ">>> 跳过 (不存在): $FOLDER"
    fi
done

echo "============================================"
echo "批量评测完成"
echo "============================================"

# 汇总所有结果
echo ""
echo "各实验结果汇总:"
echo "----------------------------------------"
printf "%-40s %12s %12s %12s\n" "Folder" "Best Pass@1" "Any Pass" "Avg Attempts"
echo "----------------------------------------"

for FOLDER in "$@"; do
    RESULT_FILE="$FOLDER/results.json"
    if [ -f "$RESULT_FILE" ]; then
        BEST_PASS=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(d['metrics']['best_pass_at_1'])")
        ANY_PASS=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(d['metrics']['any_pass'])")
        AVG_ATTEMPTS=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(d['statistics']['avg_attempts_per_question'])")
        printf "%-40s %11s%% %11s%% %12s\n" "$(basename $FOLDER)" "$BEST_PASS" "$ANY_PASS" "$AVG_ATTEMPTS"
    fi
done
echo "----------------------------------------"