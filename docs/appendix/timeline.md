# 大模型发展时间线

> 本页由 `refs/timeline.yaml` 自动生成，请勿手工编辑。
> 运行 `python scripts/gen_timeline.py` 更新。

共收录 **41** 个节点，覆盖 1943–2025 年。

## 1940s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 1943-01 | [麦卡洛克-皮茨神经元](https://doi.org/10.1007/BF02478259)<br><small>McCulloch-Pitts Neuron</small> | 🧠 理论 | 用阈值逻辑单元描述神经活动，奠定"计算单元"的抽象。 |

## 1950s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 1950-10 | [图灵测试](https://doi.org/10.1093/mind/LIX.236.433)<br><small>Computing Machinery and Intelligence</small> | 🧠 理论 | 把"机器能否思考"转化为可操作的行为判据。 |
| 1958-07 | [感知机](https://doi.org/10.1037/h0042519)<br><small>The Perceptron</small> | 🧠 理论 | 第一个可学习的线性分类器，也暴露了线性不可分的局限。 |

## 1980s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 1986-10 | [反向传播](https://doi.org/10.1038/323533a0)<br><small>Backpropagation</small> | 🧠 理论 | 用链式法则高效计算多层网络梯度，深度学习的前置条件。 |

## 1990s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 1997-11 | [长短期记忆网络](https://doi.org/10.1162/neco.1997.9.8.1735)<br><small>LSTM</small> | 🏗️ 架构 | 用门控与细胞状态缓解梯度消失，统治序列建模近二十年。 |

## 2000s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 2003-06 | [神经网络语言模型](https://www.jmlr.org/papers/v3/bengio03a.html)<br><small>A Neural Probabilistic Language Model</small> | 🏗️ 架构 | 首次用稠密词向量 + 前馈网络做语言建模，困惑度显著优于 n-gram。 |

## 2010s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 2013-01 | [word2vec](https://arxiv.org/abs/1301.3781)<br><small>Efficient Estimation of Word Representations</small> | 🏗️ 架构 | 负采样 + 层次 softmax 让词向量训练可规模化，语义算术成为现象级演示。 |
| 2014-09 | [序列到序列学习](https://arxiv.org/abs/1409.3215)<br><small>Sequence to Sequence Learning</small> | 🏗️ 架构 | 编码器-解码器结构统一了翻译等变长映射任务。 |
| 2014-09 | [加性注意力](https://arxiv.org/abs/1409.0473)<br><small>Neural Machine Translation by Jointly Learning to Align and Translate</small> | 🏗️ 架构 | 让解码器动态对齐源句片段，注意力的雏形。 |
| 2015-12 | [残差连接](https://arxiv.org/abs/1512.03385)<br><small>Deep Residual Learning</small> | 🏗️ 架构 | 恒等捷径让上百层网络可训练，成为 Transformer 的必备组件。 |
| 2017-06 | [Transformer](https://arxiv.org/abs/1706.03762)<br><small>Attention Is All You Need</small> | 🏗️ 架构 | 完全用注意力替代循环，实现训练并行化，大模型时代的起点。 |
| 2018-06 | [GPT-1](https://openai.com/research/language-unsupervised)<br><small>Improving Language Understanding by Generative Pre-Training</small> | 🏗️ 架构 | 生成式预训练 + 下游微调，确立"预训练-微调"范式。 |
| 2018-10 | [BERT](https://arxiv.org/abs/1810.04805)<br><small>Bidirectional Encoder Representations from Transformers</small> | 🏗️ 架构 | 双向掩码语言建模刷新 11 项 NLP 任务，编码器路线的高峰。 |
| 2019-02 | [GPT-2](https://openai.com/research/better-language-models)<br><small>Language Models are Unsupervised Multitask Learners</small> | 📈 规模 | 15 亿参数，零样本迁移初现，因滥用风险分阶段发布。 |
| 2019-10 | [T5](https://arxiv.org/abs/1910.10683)<br><small>Exploring the Limits of Transfer Learning</small> | 🏗️ 架构 | 把一切任务统一为"文本到文本"，系统比较预训练目标与规模。 |

## 2020s

| 时间 | 事件 | 类别 | 一句话 |
|---|---|---|---|
| 2020-01 | [缩放定律](https://arxiv.org/abs/2001.08361)<br><small>Scaling Laws for Neural Language Models</small> | 📈 规模 | 损失随参数量、数据量、算力呈幂律下降，为"堆规模"提供定量依据。 |
| 2020-01 | [Switch Transformer](https://arxiv.org/abs/2101.03961)<br><small>Switch Transformers</small> | 🏗️ 架构 | 万亿参数稀疏 MoE，验证"参数量与计算量解耦"的可行性。 |
| 2020-05 | [GPT-3](https://arxiv.org/abs/2005.14165)<br><small>Language Models are Few-Shot Learners</small> | 📈 规模 | 1750 亿参数，少样本上下文学习成为新范式。 |
| 2020-10 | [ViT](https://arxiv.org/abs/2010.11929)<br><small>An Image is Worth 16x16 Words</small> | 🏗️ 架构 | 图像切 patch 直接送进 Transformer，视觉与语言架构开始统一。 |
| 2021-01 | [CLIP](https://arxiv.org/abs/2103.00020)<br><small>Contrastive Language-Image Pre-training</small> | 🏗️ 架构 | 4 亿图文对对比学习，得到可零样本分类的共享嵌入空间。 |
| 2021-04 | [RoPE](https://arxiv.org/abs/2104.09864)<br><small>RoFormer: Rotary Position Embedding</small> | 🏗️ 架构 | 用旋转矩阵编码相对位置，成为长上下文模型的事实标准。 |
| 2021-06 | [LoRA](https://arxiv.org/abs/2106.09685)<br><small>Low-Rank Adaptation</small> | ⚙️ 系统 | 用低秩增量冻结主干，把微调显存需求降低一个数量级。 |
| 2021-07 | [Codex](https://arxiv.org/abs/2107.03374)<br><small>Evaluating LLMs Trained on Code</small> | 🚀 应用 | 代码预训练带来程序合成能力，GitHub Copilot 的技术基础。 |
| 2022-01 | [InstructGPT](https://arxiv.org/abs/2203.02155)<br><small>Training Language Models to Follow Instructions with Human Feedback</small> | 🎯 对齐 | RLHF 三阶段流程成型，1.3B 模型的人类偏好超过 175B 基座。 |
| 2022-03 | [Chinchilla](https://arxiv.org/abs/2203.15556)<br><small>Training Compute-Optimal Large Language Models</small> | 📈 规模 | 参数与数据应同比扩展，多数模型此前严重训练不足。 |
| 2022-05 | [FlashAttention](https://arxiv.org/abs/2205.14135)<br><small>Fast and Memory-Efficient Exact Attention with IO-Awareness</small> | ⚙️ 系统 | 用分块与在线 softmax 把注意力做成 IO 最优，长序列训练成为可能。 |
| 2022-11 | [ChatGPT](https://openai.com/blog/chatgpt)<br><small>ChatGPT</small> | 🚀 应用 | 对话式交互把大模型推向大众，两个月用户破亿。 |
| 2023-02 | [LLaMA](https://arxiv.org/abs/2302.13971)<br><small>Open and Efficient Foundation Language Models</small> | 🔓 开源 | 小模型 + 充分数据达到强性能，开源生态由此爆发。 |
| 2023-03 | [GPT-4](https://arxiv.org/abs/2303.08774)<br><small>GPT-4 Technical Report</small> | 📈 规模 | 多模态输入 + 大幅提升的推理与考试表现，闭源路线的标杆。 |
| 2023-05 | [DPO](https://arxiv.org/abs/2305.18290)<br><small>Direct Preference Optimization</small> | 🎯 对齐 | 把 RLHF 目标解析地转成分类损失，偏好对齐不再需要 RL 循环。 |
| 2023-07 | [Llama 2](https://arxiv.org/abs/2307.09288)<br><small>Llama 2</small> | 🔓 开源 | 开放商用许可，配套对话模型与安全报告，开源进入产品化阶段。 |
| 2023-09 | [Mistral 7B](https://arxiv.org/abs/2310.06825)<br><small>Mistral 7B</small> | 🔓 开源 | 滑窗注意力 + GQA，小模型效率标杆。 |
| 2023-12 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088)<br><small>Mixtral of Experts</small> | 🏗️ 架构 | 开源稀疏 MoE 首次在通用任务上追平更大稠密模型。 |
| 2024-02 | [Gemini 1.5](https://arxiv.org/abs/2403.05530)<br><small>Gemini 1.5</small> | 🏗️ 架构 | MoE + 百万级 token 上下文，长上下文成为竞争维度。 |
| 2024-05 | [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)<br><small>DeepSeek-V2</small> | ⚙️ 系统 | 多头潜在注意力压缩 KV cache，推理成本大幅下降。 |
| 2024-05 | [GPT-4o](https://openai.com/index/hello-gpt-4o/)<br><small>GPT-4o</small> | 🚀 应用 | 文本/视觉/语音端到端统一，交互延迟接近实时对话。 |
| 2024-09 | [o1](https://openai.com/index/learning-to-reason-with-llms/)<br><small>Learning to Reason with LLMs</small> | 🎯 对齐 | 用 RL 训练长思维链，把"测试时计算"变成可扩展的能力维度。 |
| 2024-11 | [MCP](https://www.anthropic.com/news/model-context-protocol)<br><small>Model Context Protocol</small> | ⚙️ 系统 | 统一模型与外部工具/数据源的连接方式，agent 生态的接口标准。 |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437)<br><small>DeepSeek-V3 Technical Report</small> | 🔓 开源 | 671B MoE、14.8T token 预训练，训练成本与性能比刷新认知。 |
| 2025-01 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948)<br><small>DeepSeek-R1</small> | 🎯 对齐 | 纯 RL（R1-Zero）涌现反思与长链推理，推理能力可被蒸馏到小模型。 |
| 2025-04 ⚠️ | [Qwen3](https://qwenlm.github.io/blog/qwen3/)<br><small>Qwen3</small> | 🔓 开源 | 稠密与 MoE 并行的开源家族，支持思考/非思考模式切换。 |

---

## 读时间线的三个视角

1. **架构线**：RNN（1997）→ Attention（2014）→ Transformer（2017）→ MoE（2020/2024）→ 长上下文（2024）
2. **规模线**：NNLM（2003）→ GPT-3（2020）→ Scaling Laws（2020）→ Chinchilla（2022）→ 推理期扩展（2024）
3. **对齐线**：InstructGPT（2022）→ DPO（2023）→ o1/R1（2024–2025）

三条线的交汇点，就是「能用的助手」出现的位置。
