import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. 数据准备 & 解析 (保持不变)
# ==========================================
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
# 2. 数据预处理：计算每种方法的趋势线
# ==========================================
trend_dfs = []

for method in df['Method'].unique():
    # 提取当前方法的数据
    sub_df = df[df['Method'] == method].copy()
    
    # 分箱逻辑 (与之前保持一致)
    try:
        sub_df['Bin'] = pd.qcut(sub_df['Round'], q=4, duplicates='drop')
    except:
        sub_df['Bin'] = pd.cut(sub_df['Round'], bins=3)
    
    # 计算均值
    agg_df = sub_df.groupby('Bin', observed=True)[['Round', 'Accuracy']].mean().reset_index()
    agg_df['Method'] = method # 标记方法名
    trend_dfs.append(agg_df)

# 合并所有趋势数据
df_trend = pd.concat(trend_dfs)

# ==========================================
# 3. 绘图：总图 (Summary Plot)
# ==========================================
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.figure(figsize=(10, 7)) # 设置画布大小

# 定义颜色板 (使用高对比度颜色)
palette = sns.color_palette("bright", n_colors=len(methods))

# A. 画背景散点 (原始数据) - 透明度设低一点 (alpha=0.15)
sns.scatterplot(
    data=df, 
    x='Round', 
    y='Accuracy', 
    hue='Method', 
    palette=palette,
    alpha=0.2, 
    s=50, 
    legend=False # 散点不需要图例，以免重复
)

# B. 画趋势实线 (聚合数据)
sns.lineplot(
    data=df_trend, 
    x='Round', 
    y='Accuracy', 
    hue='Method', 
    style='Method',   # 不同方法用不同线型/标记
    palette=palette,
    linewidth=3,      # 线宽加粗
    markers=True,     # 显示节点标记
    markersize=10,    # 标记大小
    dashes=False      # 全部用实线，不要虚线
)

# ==========================================
# 4. 美化与保存
# ==========================================
plt.title('Summary: Scaling Law Comparison Across Methods', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Round (Cost)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

# 调整图例位置 (放在图外，避免遮挡线条)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Method')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存
plt.savefig('summary_comparison_plot.pdf', format='pdf', bbox_inches='tight')
plt.savefig('summary_comparison_plot.png', format='png', dpi=300, bbox_inches='tight')

# plt.show()