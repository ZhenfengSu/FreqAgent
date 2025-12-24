### 熵统计方案 (Quantitative Analysis)

**核心策略：分阶段（Phase-wise）+ 分层（Layer-wise）统计**

#### 1. 数据分桶 (Data Segmentation)

在测试集上运行模型，收集生成过程中的 Attention Weights，将 Token 分为三类进行掩码（Masking）统计——即需要mask后面的所有token，以便模拟模型推理时候的情况：

- Phase A: Thinking Phase

  - 统计模型在生成 `<think>...</think>` 内容时的平均 Attention 熵。
  - *目的*：验证模型在思考时是否在进行“全局整合”（预期高熵）还是“简单复读”（预期低熵）。

- Phase B: Action Phase

  - 统计模型在生成 `<tool_call>...</tool_call>` 内容时的平均 Attention 熵。
  - **关键对比点**：这是你解释“Action 激增”的核心。
  - *假设*：在删除大部分Think 实验中，Action Phase 的熵会**显著降低**（模型不再犹豫，看到 Observation 就直接触发动作，呈现“条件反射”状）。

- Phase C: Post-Response Phase

  (收到工具结果后的反应)

  - 统计模型在看到 `<tool_response>` 后，生成下一个  <\think> Token 时的熵。

#### 2. 具体的统计指标

- 指标 1：每一个 Phase的平均熵 (Average Entropy)
  - 公式：$H=-\sum p \log p$
  - *预期*：Ablation 组在 Action Phase 的深层网络（最后几层）熵值极低。
- 指标 2：<\think>和<\tool_call>注意力分配比例 (Attention Proportion / Routing)
  - 这比单纯的熵更有说服力。计算生成 对应<\think>和<\tool_call>时(注意需要把<\think>和<\tool_call>后面的token都mask掉，以模拟推理过程)，Attention 权重落在哪里的比例，除以长度避免tool response比较长而影响了结果：
  - $Score_{obs} = \sum_{k \in Observation} A_{token, k}/len(obs)$
  - $Score_{think} = \sum_{k \in Think} A_{token, k}/len(Think)$
  - $Score_{action} = \sum_{k \in action} A_{token, k}/len(action)$

obs即指的是tool response
