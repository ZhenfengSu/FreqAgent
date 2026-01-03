#!/bin/bash

# ============================================
# 评测脚本 - 评估模型预测结果
# ============================================

# 默认配置
DEFAULT_API_BASE="http://127.0.0.1:6002/v1"
DEFAULT_API_KEY="EMPTY"
DEFAULT_MODEL="qwen2.5-72b-instruct"
DEFAULT_DATASET="gaia"
DEFAULT_WORKERS=10

# 显示帮助信息
show_help() {
    echo "用法: $0 -i <input_folder> [选项]"
    echo ""
    echo "必需参数:"
    echo "  -i, --input_folder    包含 JSONL 文件的文件夹路径"
    echo ""
    echo "可选参数:"
    echo "  -d, --dataset         数据集类型 (默认: gaia)"
    echo "                        可选: gaia, browsecomp_zh, browsecomp_en, simple_qa, time_qa, webwalker, hle 等"
    echo "  -a, --api_base        API 服务地址 (默认: $DEFAULT_API_BASE)"
    echo "  -k, --api_key         API 密钥 (默认: EMPTY)"
    echo "  -m, --model           评判模型名称 (默认: $DEFAULT_MODEL)"
    echo "  -w, --workers         并发线程数 (默认: $DEFAULT_WORKERS)"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i ./predictions/experiment1"
    echo "  $0 -i ./predictions/experiment1 -d browsecomp_zh -w 20"
    echo "  $0 -i ./predictions/experiment1 -a http://localhost:8000/v1 -m gpt-4"
    echo ""
    echo "输出:"
    echo "  评测结果将保存到 <input_folder>/results.json"
}

# 解析命令行参数
INPUT_FOLDER=""
DATASET=$DEFAULT_DATASET
API_BASE=$DEFAULT_API_BASE
API_KEY=$DEFAULT_API_KEY
MODEL=$DEFAULT_MODEL
WORKERS=$DEFAULT_WORKERS

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -a|--api_base)
            API_BASE="$2"
            shift 2
            ;;
        -k|--api_key)
            API_KEY="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$INPUT_FOLDER" ]; then
    echo "错误: 必须指定输入文件夹 (-i)"
    echo ""
    show_help
    exit 1
fi

# 检查文件夹是否存在
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "错误: 文件夹不存在: $INPUT_FOLDER"
    exit 1
fi

# 检查文件夹中是否有 JSONL 文件
JSONL_COUNT=$(find "$INPUT_FOLDER" -maxdepth 1 -name "*.jsonl" | wc -l)
if [ "$JSONL_COUNT" -eq 0 ]; then
    echo "错误: 文件夹中没有找到 JSONL 文件: $INPUT_FOLDER"
    exit 1
fi

echo "============================================"
echo "开始评测"
echo "============================================"
echo "输入文件夹: $INPUT_FOLDER"
echo "数据集类型: $DATASET"
echo "API 地址: $API_BASE"
echo "评判模型: $MODEL"
echo "并发线程: $WORKERS"
echo "JSONL 文件数: $JSONL_COUNT"
echo "============================================"
echo ""

# 运行评测脚本
python evaluate.py \
    --input_folder "$INPUT_FOLDER" \
    --dataset "$DATASET" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL" \
    --max_workers "$WORKERS"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "评测完成!"
    echo "结果已保存到: $INPUT_FOLDER/results.json"
    echo "============================================"
else
    echo ""
    echo "评测失败，请检查错误信息"
    exit 1
fi