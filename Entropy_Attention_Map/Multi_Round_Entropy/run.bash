# 绘制第一个 case（默认）
python script.py -i data.json --plot

# 绘制指定 case 并保存
python script.py -i data.json --plot --case 5 --save-plot case5.png

# 交互式查看多个 case
python script.py -i data.json --interactive

# 显示 token 文本（适合 token 数较少的情况）
python script.py -i data.json --plot --case 0 --show-tokens