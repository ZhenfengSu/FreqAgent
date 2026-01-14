import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 数据准备
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

# 2. 数据解析
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

# ==========================================
# 3. 定义绘图函数 (核心修改)
# ==========================================
def plot_binned_trend(data, color=None, label=None, **kwargs):
    """
    这个函数会接收当前子图的所有数据 (data)，
    1. 画原始散点
    2. 计算分箱平均
    3. 画平均线
    """
    ax = plt.gca() # 获取当前子图对象
    
    # --- A. 画原始散点 (背景，半透明) ---
    sns.scatterplot(
        data=data, x='Round', y='Accuracy', 
        color=color, alpha=0.3, s=40, edgecolor=None, ax=ax
    )
    
    # --- B. 计算分箱平均 (Binning) ---
    # 将数据按 Round 大小切分成 4 个区间 (qcut 保证每个箱子里的点数大致相同，cut 保证区间长度相同)
    # 这里使用 qcut (Quantile Cut) 防止某个区间没有点导致断线
    try:
        # 尝试切分 4 份，如果数据点太少切不了，就切 3 份
        data['Bin'] = pd.qcut(data['Round'], q=4, duplicates='drop')
    except:
        data['Bin'] = pd.cut(data['Round'], bins=3)
        
    # 计算每个箱子的平均值
    trend_data = data.groupby('Bin', observed=True).agg({
        'Round': 'mean', 
        'Accuracy': 'mean'
    }).reset_index()
    
    # --- C. 画趋势实线 ---
    sns.lineplot(
        data=trend_data, x='Round', y='Accuracy',
        color=color, linewidth=2.5, marker='o', markersize=8, ax=ax
    )

# ==========================================
# 4. 绘图执行
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

g = sns.FacetGrid(
    df, 
    col="Method", 
    col_wrap=3, 
    height=3.5, 
    aspect=1.2, 
    sharex=False, 
    sharey=True
)

# 使用 map_dataframe 确保数据正确传递
g.map_dataframe(plot_binned_trend, color="#2b579a") # 使用深蓝色

# 5. 美化
g.fig.suptitle('Scaling Law Analysis: Binned Average Trends', fontsize=16, fontweight='bold', y=1.02)

for ax, title in zip(g.axes.flat, methods):
    # 清理标题
    readable_title = title.replace('_', ' ').title()
    ax.set_title(readable_title, fontsize=11, fontweight='bold')
    
    # 强制显示网格
    ax.grid(True, linestyle='--', alpha=0.5)

g.set_axis_labels("Round (Cost)", "Accuracy (%)")

plt.tight_layout()
# plt.show()
plt.savefig('method_accuracy_round_smoothed_lmplot.pdf', format='pdf', bbox_inches='tight') # 建议保存为PDF矢量图
plt.savefig('method_accuracy_round_smoothed_lmplot.png', format='png', dpi=300, bbox_inches='tight') # 也可以保存为高分辨率PNG