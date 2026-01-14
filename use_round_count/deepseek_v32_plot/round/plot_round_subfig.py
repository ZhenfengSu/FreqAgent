import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 数据准备 (保持不变)
data_str = """
Method baseline filter_think filter_keep_last_think fiter_keep_last_tool long_context_filter_tool_p75 long_context_filter_tool_p100
acc 42.72 47.57 48.54 39.81 46.6 44.66
round 561 710 539 536 568 538
acc 45.63 44.66 45.63 41.75 41.75 47.57
round 559 681 547 538 536 545
acc 43.69 49.51 46.6 36.89 41.75 41.75
round 575 731 538 571 579 574
acc 44.66 46.6 43.69 36.89 43.69 46.6
round 601 868 676 761 651 649
acc 49.51 43.69 44.66 38.83 47.57 47.57
round 606 904 716 815 631 615
acc 45.63 45.63 51.46 36.89 42.72 45.63
round 606 918 621 777 686 645
acc 44.66 49.51 49.51 32.04 52.43 43.69
round 674 875 694 1088 726 732
acc 43.69 52.43 53.4 35.92 50.49 54.37
round 682 1010 633 1214 653 654
acc 42.72 52.43 45.63 36.89 47.57 49.51
round 730 940 739 923 756 749
acc 42.72 51.46 49.51 41.75 48.54 43.69
round 659 947 708 1542 923 823
acc 45.63 49.51 44.66 36.89 45.63 40.78
round 589 911 732 1696 929 859
acc 42.72 48.54 51.46 40.78 50.49 44.66
round 699 1072 711 1483 792 720
"""

# 2. 数据解析 (保持不变)
lines = data_str.strip().split('\n')
header = lines[0].split()
methods = header[1:]
data_list = []

for i in range(1, len(lines), 2):
    acc_line = lines[i].split()
    round_line = lines[i+1].split()
    if acc_line[0] == 'acc' and round_line[0] == 'round':
        acc_values = [float(x) for x in acc_line[1:]]
        round_values = [float(x) for x in round_line[1:]]
        for method, acc, rnd in zip(methods, acc_values, round_values):
            data_list.append({'Method': method, 'Accuracy': acc, 'Round': rnd})

df = pd.DataFrame(data_list)

# 3. 绘图设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'

# 使用 lmplot 创建分面网格图
# col="Method": 按照 Method 分列
# col_wrap=3: 每行显示3个图，自动换行
# sharex=False: 允许每个子图有独立的X轴范围（因为有些方法的Round很大，有些很小，独立轴能看清局部趋势）
# sharey=True: 共享Y轴范围（方便对比不同方法的绝对精度高低）
g = sns.lmplot(
    data=df, 
    x="Round", 
    y="Accuracy", 
    col="Method", 
    col_wrap=3, 
    height=3.5,    # 每个子图的高度
    aspect=1.2,    # 宽高比
    hue="Method",  # 颜色区分
    palette="deep",
    ci=95,         # 显示95%置信区间（阴影部分）
    scatter_kws={"s": 50, "alpha": 0.7, "edgecolor": "white"}, # 散点样式
    line_kws={"linewidth": 2}, # 回归线样式
    sharex=False,  # 关键：设为False以便看清每个方法内部的分布，设为True则便于横向对比Round大小
    sharey=True    # 关键：设为True以便对比精度高低
)

# 4. 图表美化
# 设置总标题
g.fig.suptitle('Trend Analysis: Accuracy vs. Round by Strategy', fontsize=16, fontweight='bold', y=1.02)

# 遍历每个子图进行微调
for ax, title in zip(g.axes.flat, methods):
    # 替换标题中的下划线，使其更美观
    readable_title = title.replace('_', ' ').title()
    ax.set_title(readable_title, fontsize=12, fontweight='bold')
    
    # 确保网格线存在
    ax.grid(True, linestyle='--', alpha=0.5)

# 设置轴标签
g.set_axis_labels("Round (Cost)", "Accuracy (%)")

plt.tight_layout()
plt.savefig('method_accuracy_round_lmplot.pdf', format='pdf', bbox_inches='tight') # 建议保存为PDF矢量图
plt.savefig('method_accuracy_round_lmplot.png', format='png', dpi=300, bbox_inches='tight') # 也可以保存为高分辨率PNG