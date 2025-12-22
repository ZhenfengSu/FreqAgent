### **最终实验方案：基于频域分析的大模型思维与工具调用模式研究**

#### **1. 数据获取与推理 (Data Acquisition & Inference)**

我们将构建包含 $N$ 个不同任务指令的数据集，将其输入到目标大模型中进行推理。

- **提取对象**：针对每一个输入样本，我们不局限于最后一层，而是提取模型**中后层**（例如 Llama3-70B 的第 60-70 层，或 32 层模型的第 24-28 层）的 **Key (K)** 和 **Value (V)** 缓存矩阵。
- **数据形态**：对于每个样本，提取出的矩阵形状为 `[Seq_Len, Num_Heads, Head_Dim]`。

#### **2. 语义切片 (Semantic Slicing)**

利用特殊 Token ID 或文本解析方法，定位序列中不同语义角色的起止位置，将完整的 KV 序列切割为三类独立的张量片段：

1. **思维链片段 (`<think>`)**：模型内部推理过程。
2. **工具调用片段 (`<tool_call>`)**：模型生成调用代码或指令的部分。
3. **工具响应片段 (`<tool_response>`)**：外部工具返回的结果部分。

#### **3. 时域预处理 (Time-Domain Preprocessing)**

为了确保不同长度的片段在频域上具有可比性（即实现“频桶对齐”），我们对切片后的张量执行以下两步关键操作：

- 固定长度重采样 (Fixed-Length Resampling)：设定一个统一的目标长度 $L_{target}$（例如 $L=128$ 或 $256$）。利用线性插值（Linear Interpolation）算法，将所有长短不一的片段强制缩放至该固定长度。
  - *目的*：确保变换后的频谱图中，第 $k$ 个数据点对所有样本都代表相同的物理频率。
- Z-Score 归一化 (Normalization)：对每个重采样后的片段进行标准化处理，即减去均值并除以标准差 $(x - \mu) / \sigma$
  - *目的*：消除不同层或不同样本间的绝对幅值差异，专注于信号的波动模式（纹理特征）。

#### **4. 频域变换 (Frequency Domain Transformation)**

对预处理后的长度为 $L_{target}$ 的序列进行**离散余弦变换 (DCT)** 或快速傅里叶变换 (FFT)。

- **计算功率谱**：取变换结果的幅值平方，得到功率谱密度（Power Spectrum Density）。
- 这一步将时域上的数值波动转化为频域上的能量分布，揭示信号在低频（长程依赖/平滑语义）与高频（突变/噪声/特定触发）上的特征。

#### **5. 统计聚合与对比 (Aggregation & Comparison)**

- **聚合**：分别对 `<think>`、`<tool_call>`、`<tool_response>` 三类数据的所有样本、所有注意力头（Heads）的频谱进行平均，得到三条标准的**平均频谱曲线**。
- 对比分析：
  - 观察 **`<think>`** 是否表现出类似于自然语言的平滑低频特征（代表连贯推理）。
  - 观察 **`<tool_call>`** 是否在特定高频段存在能量峰值（代表结构化、刚性的语法模式）。
  - 观察 **`<tool_response>`** 的频谱特征是更接近自然语言，还是表现出高频噪声（代表数据密集型内容）。


------

### 最终方案流程图

为了让你在写论文或报告时更清晰，我为你生成了一个基于 HTML/CSS 的流程可视化图表。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --primary: #4a90e2;
            --secondary: #50c878;
            --accent: #ff6b6b;
            --text: #333333;
            --arrow: #999;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text);
            padding: 20px;
            line-height: 1.6;
        }

        .pipeline-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .step {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid var(--primary);
            position: relative;
        }

        .step h3 {
            margin-top: 0;
            color: var(--primary);
            font-size: 1.1rem;
        }

        .step p {
            margin-bottom: 0;
            font-size: 0.95rem;
            color: #555;
        }

        .arrow-down {
            text-align: center;
            color: var(--arrow);
            font-size: 1.5rem;
            margin: -10px 0;
        }

        .branch-container {
            display: flex;
            gap: 10px;
            justify-content: space-between;
        }

        .branch {
            flex: 1;
            background: #fff;
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            font-size: 0.9rem;
        }

        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            color: white;
            margin-bottom: 5px;
        }

        .tag.think { background-color: var(--primary); }
        .tag.tool { background-color: var(--secondary); }
        .tag.resp { background-color: var(--accent); }

        .note {
            font-size: 0.85rem;
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }

        @media (max-width: 600px) {
            .branch-container {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>

<div class="pipeline-container">
    <h2 style="text-align: center; color: #333;">KV Cache 频域分析流水线</h2>

    <!-- Step 1 -->
    <div class="step">
        <h3>1. 模型推理 (Inference)</h3>
        <p>输入 N 个 Prompt，运行大模型。捕获特定层（如 Layer 28）的 KV Cache。</p>
        <div class="note">数据形状: [Batch, Seq_Len, Head, Head_Dim]</div>
    </div>

    <div class="arrow-down">↓</div>

    <!-- Step 2 -->
    <div class="step">
        <h3>2. 语义切片 (Slicing)</h3>
        <p>根据 Token ID 或特殊标记，将序列物理切割为独立片段。</p>
        <div class="branch-container" style="margin-top:10px;">
            <div class="branch">
                <span class="tag think">&lt;think&gt;</span><br>
                提取所有思考片段<br>
                (可能多个)
            </div>
            <div class="branch">
                <span class="tag tool">&lt;tool_call&gt;</span><br>
                提取工具调用片段
            </div>
            <div class="branch">
                <span class="tag resp">&lt;tool_resp&gt;</span><br>
                提取工具返回结果
            </div>
        </div>
    </div>

    <div class="arrow-down">↓</div>

    <!-- Step 3 -->
    <div class="step" style="border-left-color: #e67e22;">
        <h3>3. 预处理与重采样 (Preprocessing)</h3>
        <p><b>关键步骤：</b>将所有不同长度的片段，插值(Interpolate)到固定长度。</p>
        <p>例如：全部缩放至 <b>L = 128</b>。</p>
        <div class="note">目的：确保频域变换后的横坐标（频率桶）物理意义一致。</div>
    </div>

    <div class="arrow-down">↓</div>

    <!-- Step 4 -->
    <div class="step">
        <h3>4. 归一化 (Normalization)</h3>
        <p>对每个片段执行 <code>(x - mean) / std</code>。</p>
        <p>消除幅值差异，仅保留波动模式。</p>
    </div>

    <div class="arrow-down">↓</div>

    <!-- Step 5 -->
    <div class="step" style="border-left-color: #9b59b6;">
        <h3>5. 频域变换 (DCT)</h3>
        <p>对每个长度为 128 的片段进行离散余弦变换。</p>
        <p>计算功率谱 <code>Power = DCT(x)^2</code>。</p>
    </div>

    <div class="arrow-down">↓</div>

    <!-- Step 6 -->
    <div class="step">
        <h3>6. 聚合与对比 (Aggregation)</h3>
        <p>分别计算三类标签的平均频谱。</p>
        <div class="branch-container" style="margin-top:10px;">
            <div class="branch" style="border: 2px solid var(--primary);">
                <b>Think 均值谱</b><br>
                观察：是否有高频噪点？
            </div>
            <div class="branch" style="border: 2px solid var(--secondary);">
                <b>Tool 均值谱</b><br>
                观察：是否特定频段突出？
            </div>
            <div class="branch" style="border: 2px solid var(--accent);">
                <b>Resp 均值谱</b><br>
                观察：是否接近自然语言？
            </div>
        </div>
    </div>

</div>

</body>
</html>
```

### 备注：DCT采用下面的算法

The Discrete Cosine Transform (DCT) transforms a signal from the spatial domain (time or position) into the frequency domain. Several variants of DCT exist, with DCT-II being the most common. For a real-value discrete signal $X_{0:N-1} = [x_0, \ldots, x_{N-1}]$ of length $N$, it is defined as:
$$y_t = \alpha_t \sum_{n=0}^{N-1} x_n \cdot \cos \left[ \frac{\pi t(2n+1)}{2N} \right], \quad \alpha_t = \begin{cases} \sqrt{\frac{1}{N}} & \text{if } t = 0, \\ \sqrt{\frac{2}{N}} & \text{otherwise} \end{cases}$$
where $t = 0, 1, \cdots, N-1$. $\alpha_t$ is the normalization factor.
The time-domain signal $X_{0:N-1}$ can be recovered by applying the inverse DCT (IDCT) on the frequency components $Y_{0:N-1}$:
$$
x_n = \sum_{t=0}^{N-1} \alpha_t \cdot y_t \cdot \cos \left[ \frac{\pi t(2n+1)}{2N} \right].$$
The frequency components are expressed as a combination of the original signals. The values can be computed using the Fast Fourier Transform (FFT) with a complexity of $O(N \log N)$. The amplitudes of frequency components are utilized in the power spectrum analysis to represent the energy or magnitude of components. The components of higher energy in the frequency domain indicate they are more informative (He et al., 2023).

