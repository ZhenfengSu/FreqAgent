import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. 准备数据
# 这里直接使用了你提供的文本数据
data_str = """method	baseline	filter_think	filter_keep_last_think	fiter_keep_last_tool	long_context_filter_tool_p75	long_context_filter_tool_p100
acc	44.66	47.57	48.54	39.81	46.6	44.66
time	2951.534605	2930.129681	2387.596685	1927.570757	2103.985866	2783.218787
acc	49.51	44.66	45.63	41.75	41.75	47.57
time	2922.996927	2808.928813	2464.841051	1747.022303	2698.222584	2628.430622
acc	45.63	49.51	46.6	36.89	41.75	41.75
time	2728.52614	3062.362899	2600.581574	1724.967573	2603.544439	3186.810009
acc	44.66	46.6	43.69	36.89	43.69	46.6
time	3422.274695	3727.040281	4712.244223	1911.641906	2918.981355	9611.224658
acc	43.69	43.69	44.66	38.83	47.57	47.57
time	2776.832864	3639.457614	3288.935232	2155.683576	2790.660164	2664.613821
acc	42.72	45.63	51.46	36.89	42.72	45.63
time	3153.408492	4071.836239	2974.603683	2068.51388	3044.448772	2974.115544
acc	42.72	49.51	49.51	32.04	52.43	43.69
time	2228.253992	3185.035951	3614.755446	1947.002292	2950.038482	2889.785569
acc	45.63	52.43	53.4	35.92	50.49	54.37
time	2097.866442	3409.488782	2807.727212	2137.016837	2964.425746	2770.3282
acc	42.72	52.43	45.63	36.89	47.57	49.51
time	2811.917454	3542.054467	3517.320791	1936.128347	3660.731929	3256.789419"""

# 读取数据，假设列之间是用制表符分隔的
df_raw = pd.read_csv(io.StringIO(data_str), sep='\t')

# 2. 数据清洗与重构
# 原始数据是交错的：一行acc，一行time。我们需要将其转换为 (Method, Acc, Time) 的格式
clean_data = []

# 获取所有方法名称（排除第一列 'method' 或 'Unnamed: 0'）
methods = df_raw.columns[1:] 

# 遍历每一列（每种方法）
for method in methods:
    # 提取该列的所有数据
    col_data = df_raw[method].values
    
    # 步长为2进行遍历，每次取 i (Acc) 和 i+1 (Time)
    for i in range(0, len(col_data), 2):
        if i + 1 < len(col_data):
            acc_val = float(col_data[i])
            time_val = float(col_data[i+1])
            
            clean_data.append({
                'Method': method,
                'Accuracy (%)': acc_val,
                'Time (ms)': time_val
            })

df_plot = pd.DataFrame(clean_data)

# 3. 绘图
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# 绘制散点图
# hue: 颜色区分方法
# style: 形状区分方法（可选，增加区分度）
# s: 点的大小
scatter = sns.scatterplot(
    data=df_plot, 
    x='Time (ms)', 
    y='Accuracy (%)', 
    hue='Method', 
    style='Method', 
    s=120, 
    alpha=0.8
)

# 计算每个方法的平均值并标记（可选，帮助看中心趋势）
means = df_plot.groupby('Method')[['Time (ms)', 'Accuracy (%)']].mean().reset_index()
for index, row in means.iterrows():
    plt.text(
        row['Time (ms)'], 
        row['Accuracy (%)'] + 0.5, 
        f"{row['Method']}", 
        horizontalalignment='center', 
        size='small', 
        color='black', 
        weight='bold'
    )
    # 绘制平均点（用大一点的星星表示）
    plt.scatter(row['Time (ms)'], row['Accuracy (%)'], marker='*', s=200, color='red', edgecolors='black', zorder=10, label='Mean' if index==0 else "")

# 4. 图表美化
plt.title('DeepSeek-V3 Agent Strategy Analysis: Accuracy vs Time Scaling', fontsize=16)
plt.xlabel('Time Cost (ms)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 调整布局防止图例被切掉
plt.tight_layout()

# 显示图表
plt.savefig('deepseek_v32_strategy_analysis.png', dpi=300)

# 5. 简单的统计输出
print("各策略平均表现 (按Accuracy降序排列):")
print(means.sort_values('Accuracy (%)', ascending=False).to_string(index=False))