import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# 1. 准备数据
# 将用户提供的文本数据模拟为 CSV 格式字符串
data_str = """
Method	baseline	filter_think	filter_keep_last_think	filter_keep_last_tool	long_context_filter_tool_p75	long_context_filter_tool_p100
acc	42.72	47.57	48.54	39.81	46.6	44.66
time	2572.667047	2930.129681	2387.596685	1927.570757	2103.985866	2783.218787
acc	45.63	44.66	45.63	41.75	41.75	47.57
time	2412.759609	2808.928813	2464.841051	1747.022303	2698.222584	2628.430622
acc	43.69	49.51	46.6	36.89	41.75	41.75
time	2698.723295	3062.362899	2600.581574	1724.967573	2603.544439	3186.810009
acc	44.66	46.6	43.69	36.89	43.69	46.6
time	2951.534605	3727.040281	4712.244223	1911.641906	2918.981355	9611.224658
acc	49.51	43.69	44.66	38.83	47.57	47.57
time	2922.996927	3639.457614	3288.935232	2155.683576	2790.660164	2664.613821
acc	45.63	45.63	51.46	36.89	42.72	45.63
time	2728.52614	4071.836239	2974.603683	2068.51388	3044.448772	2974.115544
acc	44.66	49.51	49.51	32.04	52.43	43.69
time	3422.274695	3185.035951	3614.755446	1947.002292	2950.038482	2889.785569
acc	43.69	52.43	53.4	35.92	50.49	54.37
time	2776.832864	3409.488782	2807.727212	2137.016837	2964.425746	2770.3282
acc	42.72	52.43	45.63	36.89	47.57	49.51
time	3153.408492	3542.054467	3517.320791	1936.128347	3660.731929	3256.789419
acc	42.72	51.46	49.51	41.75	48.54	43.69
time	2228.253992	3271.009875	2757.871256	2463.221735	6573.44356	5761.898254
acc	45.63	49.51	44.66	36.89	45.63	40.78
time	2097.866442	4060.072174	5233.408085	2402.578008	3132.786402	5079.07951
acc	42.72	48.54	51.46	40.78	50.49	44.66
time	2811.917454	3019.665303	3026.549834	2297.788677	3345.647926	2933.660902
"""

# 读取数据
# 注意：原数据第一列是 acc, time 交替，且第一行是表头
df_raw = pd.read_csv(io.StringIO(data_str), sep='\t')

# 2. 数据清洗与重组
# 目标是将数据转换为长格式：[Method, Time, Accuracy]

methods = df_raw.columns[1:] # 获取所有方法名（跳过第一列 'Method'）
plot_data = []

# 遍历每一列（每种方法）
for method in methods:
    col_data = df_raw[method].values
    # 数据是成对的：偶数索引是 acc，奇数索引是 time
    acc_values = col_data[0::2].astype(float)
    time_values = col_data[1::2].astype(float)
    
    for t, a in zip(time_values, acc_values):
        plot_data.append({
            'Method': method,
            'Time': t,
            'Accuracy': a
        })

df_clean = pd.DataFrame(plot_data)

# 3. 绘图
plt.figure(figsize=(12, 8))

# 为每个方法绘制散点
# 使用 tab10 调色板以获得清晰的区分
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

for i, method in enumerate(methods):
    subset = df_clean[df_clean['Method'] == method]
    plt.scatter(
        subset['Time'], 
        subset['Accuracy'], 
        label=method, 
        s=100,      # 点的大小
        alpha=0.7,  # 透明度
        color=colors[i]
    )
    
    # 可选：添加趋势线（线性拟合）看 scaling 趋势
    z = np.polyfit(subset['Time'], subset['Accuracy'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(subset['Time'].min(), subset['Time'].max(), 100)
    plt.plot(x_trend, p(x_trend), linestyle='--', alpha=0.5, color=colors[i])

# 4. 图表美化
plt.title('Accuracy vs. Time Scaling by Agent Strategy (DeepSeek V3)', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整布局以防止图例被切断
plt.tight_layout()

# 显示图表
# plt.show()
# 保存图表
plt.savefig('deepseek_v32_time_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('deepseek_v32_time_accuracy_plot.pdf', bbox_inches='tight')