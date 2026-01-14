import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import math

# 1. 准备数据
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
df_raw = pd.read_csv(io.StringIO(data_str), sep='\t')

# 2. 数据处理
methods = df_raw.columns[1:] # 获取所有方法名
plot_data = {} # 使用字典存储每种方法的数据

# 找出全局的最大最小值，统一坐标轴范围，方便对比
all_times = []
all_accs = []

for method in methods:
    col_data = df_raw[method].values
    acc_values = col_data[0::2].astype(float)
    time_values = col_data[1::2].astype(float)
    
    plot_data[method] = {
        'time': time_values,
        'acc': acc_values
    }
    all_times.extend(time_values)
    all_accs.extend(acc_values)

# 计算坐标轴范围（留一点边距）
x_min, x_max = min(all_times) * 0.9, max(all_times) * 1.05
y_min, y_max = min(all_accs) * 0.95, max(all_accs) * 1.05

# 3. 绘制六个子图 (2行3列)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten() # 将二维数组展平，方便遍历

# 颜色列表
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

for i, method in enumerate(methods):
    ax = axes[i]
    data = plot_data[method]
    
    # 绘制散点
    ax.scatter(data['time'], data['acc'], color=colors[i], s=80, alpha=0.7, label='Data Points')
    
    # 绘制趋势线 (一次线性拟合)
    if len(data['time']) > 1:
        z = np.polyfit(data['time'], data['acc'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(data['time']), max(data['time']), 100)
        ax.plot(x_trend, p(x_trend), linestyle='--', color='gray', alpha=0.8, label='Trend')
        
        # 在标题中显示斜率，斜率正表示时间越长精度越高
        slope = z[0] * 1000 # 放大斜率数值方便阅读 (每1000ms提升多少acc)
        title_suffix = f"(Slope: {slope:.2f}/s)"
    else:
        title_suffix = ""

    # 设置子图样式
    ax.set_title(f"{method}\n{title_suffix}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy (%)')
    
    # 统一坐标轴范围，这样可以直接横向对比不同图
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.grid(True, linestyle=':', alpha=0.6)

# 调整整体布局
plt.tight_layout()
plt.suptitle('Accuracy vs. Time Scaling per Strategy (DeepSeek V3)', fontsize=16, y=1.02)

# plt.show()
plt.savefig('deepseek_v32_time_accuracy_subfig.png', dpi=300, bbox_inches='tight')
plt.savefig('deepseek_v32_time_accuracy_subfig.pdf', bbox_inches='tight')