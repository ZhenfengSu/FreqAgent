import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# 1. 数据准备
# 将您提供的文本数据模拟为一个字符串流
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
# 读取原始数据
lines = data_str.strip().split('\n')
header = lines[0].split()
methods = header[1:] # 获取所有方法名

data_list = []

# 遍历每一对 acc 和 round 行
# 从第1行开始，每次步进2行
for i in range(1, len(lines), 2):
    acc_line = lines[i].split()
    round_line = lines[i+1].split()
    
    # 确保是 acc 和 round 行
    if acc_line[0] == 'acc' and round_line[0] == 'round':
        acc_values = [float(x) for x in acc_line[1:]]
        round_values = [float(x) for x in round_line[1:]]
        
        # 将每个方法的数据添加到列表中
        for method, acc, rnd in zip(methods, acc_values, round_values):
            data_list.append({
                'Method': method,
                'Accuracy': acc,
                'Round': rnd
            })

df = pd.DataFrame(data_list)

# 3. 设置学术绘图风格
# 使用 seaborn 的 paper context 和 whitegrid style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif' # 设置无衬线字体，通常学术图表首选 Arial 或 Helvetica
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 创建画布
plt.figure(figsize=(10, 6), dpi=300)

# 定义标记形状，以便在黑白打印时也能区分
markers = ['o', 's', '^', 'D', 'v', 'X'] 
# 定义颜色调色板 (Colorblind friendly is better for academic)
palette = sns.color_palette("deep")

# 4. 绘制散点图与回归趋势线
# lmplot 是 seaborn 中用于绘制线性回归模型的强力工具，但为了更好的自定义控制，我们分开画
# 这里使用 scatterplot 画点，regplot 画线太乱，我们用 lmplot 的逻辑手动循环或者直接用 scatterplot + 趋势线

# 绘制散点
scatter = sns.scatterplot(
    data=df, 
    x='Round', 
    y='Accuracy', 
    hue='Method', 
    style='Method',
    markers=markers[:len(methods)],
    palette=palette[:len(methods)],
    s=80, # 点的大小
    alpha=0.8, # 透明度
    edgecolor='k', # 点的边缘颜色
    linewidth=0.5
)

# 可选：添加简单的线性拟合线来观察趋势 (如果不想要趋势线，可以注释掉这一段)
# 这一段会为每个方法画一条淡淡的趋势线
colors = palette[:len(methods)]
for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    # 使用 numpy 进行简单的线性拟合 (1次多项式)
    if len(subset) > 1:
        z = np.polyfit(subset['Round'], subset['Accuracy'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(subset['Round'].min(), subset['Round'].max(), 100)
        plt.plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.5, linewidth=1.5)

# 5. 图表美化与标注
plt.title('Accuracy vs. Round Consumption by Agent Strategy', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Round (Computational Cost)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# 优化图例
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Method', frameon=True, shadow=False)

# 设置轴范围（根据数据自动调整，稍微留点边距）
plt.xlim(df['Round'].min() * 0.9, df['Round'].max() * 1.05)
plt.ylim(df['Accuracy'].min() * 0.9, df['Accuracy'].max() * 1.05)

# 添加网格线 (通常学术图表保留横向网格线辅助读数，纵向可选)
plt.grid(True, linestyle='--', alpha=0.6)

# 紧凑布局
plt.tight_layout()

# 6. 显示或保存
plt.savefig('agent_accuracy_round_scatter.pdf', format='pdf', bbox_inches='tight') # 建议保存为PDF矢量图
plt.savefig('agent_accuracy_round_scatter.png', format='png', dpi=300, bbox_inches='tight') # 也可以保存为高分辨率PNG