# 论文清单

> 按主题分组，标注**必读**（🔴）与**选读**（⚪）。一句话说明"它解决了什么问题"，而不是"它做了什么"。
> 链接会由 `make check` 定期校验；失效链接请在 issue 中报告。

## 一、前史与基础

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2003 | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) | 首次用稠密词向量 + 神经网络做语言建模 | 🔴 |
| 2013 | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) | word2vec：让词向量训练规模化 | 🔴 |
| 2014 | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | 编码器-解码器统一变长映射 | 🔴 |
| 2014 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 加性注意力，注意力的起点 | 🔴 |
| 1997 | [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) | 门控机制解决梯度消失 | ⚪ |

## 二、Transformer 与预训练

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 用注意力替代循环，训练可并行 | 🔴 |
| 2018 | [BERT](https://arxiv.org/abs/1810.04805) | 双向掩码预训练，编码器路线的高峰 | 🔴 |
| 2019 | [T5](https://arxiv.org/abs/1910.10683) | 一切任务统一为文本到文本 | ⚪ |
| 2019 | [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) | 多头分工不均，可剪枝 | ⚪ |
| 2020 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 损失随规模幂律下降 | 🔴 |
| 2020 | [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 上下文学习成为新范式 | 🔴 |
| 2020 | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 图像切 patch 进 Transformer | 🔴 |
| 2022 | [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | 参数与数据应同比扩展 | 🔴 |

## 三、位置编码与注意力优化

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2021 | [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 旋转编码相对位置 | 🔴 |
| 2019 | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | MQA，KV cache 优化的起点 | ⚪ |
| 2022 | [FlashAttention](https://arxiv.org/abs/2205.14135) | IO 感知的精确注意力 | 🔴 |
| 2023 | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | 分组查询注意力 | 🔴 |
| 2023 | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | 注意力汇与流式推理 | ⚪ |
| 2023 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | 线性复杂度状态空间模型 | ⚪ |

## 四、对齐与偏好优化

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2022 | [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | RLHF 三阶段流程成型 | 🔴 |
| 2023 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 把 RLHF 目标转成分类损失 | 🔴 |
| 2024 | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) | 组内相对优势，去掉价值网络 | 🔴 |
| 2023 | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 小模型 + 充分数据 | 🔴 |
| 2023 | [Llama 2](https://arxiv.org/abs/2307.09288) | 开放商用 + 安全报告 | ⚪ |
| 2024 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 低成本 MoE 预训练的工程范本 | 🔴 |
| 2025 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 纯 RL 涌现长链推理，能力可蒸馏 | 🔴 |

## 五、架构与系统

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2021 | [Switch Transformers](https://arxiv.org/abs/2101.03961) | 万亿参数稀疏 MoE | ⚪ |
| 2024 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 细粒度专家 + 共享专家 | 🔴 |
| 2024 | [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434) | 多头潜在注意力压缩 KV cache | 🔴 |
| 2023 | [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180) | 分页管理 KV cache | 🔴 |
| 2021 | [LoRA](https://arxiv.org/abs/2106.09685) | 低秩适配，微调显存降一个数量级 | 🔴 |
| 2023 | [QLoRA](https://arxiv.org/abs/2305.14314) | 4bit 量化 + LoRA | ⚪ |
| 2022 | [GPTQ](https://arxiv.org/abs/2210.17323) | 一次性权重量化 | ⚪ |
| 2023 | [AWQ](https://arxiv.org/abs/2306.00978) | 激活感知的权重量化 | ⚪ |

## 六、应用层

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2020 | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | RAG 的原始形式 | 🔴 |
| 2020 | [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) | 双塔稠密检索 | ⚪ |
| 2022 | [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 显式中间步骤提升推理 | 🔴 |
| 2022 | [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) | 多路径投票 | 🔴 |
| 2022 | [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) | 推理与行动交替 | 🔴 |
| 2023 | [Reflexion](https://arxiv.org/abs/2303.11366) | 语言化反思改进后续尝试 | 🔴 |
| 2023 | [Toolformer](https://arxiv.org/abs/2302.04761) | 模型自学调用 API | ⚪ |
| 2023 | [Generative Agents](https://arxiv.org/abs/2304.03442) | 记忆流 + 检索打分 | 🔴 |
| 2023 | [MemGPT](https://arxiv.org/abs/2310.08560) | 上下文分页 | 🔴 |
| 2023 | [Voyager](https://arxiv.org/abs/2305.16291) | 技能库 + 自动课程 | ⚪ |
| 2023 | [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | 自我纠错的反证 | 🔴 |
| 2021 | [CLIP](https://arxiv.org/abs/2103.00020) | 图文对比学习 | 🔴 |
| 2020 | [DDPM](https://arxiv.org/abs/2006.11239) | 扩散模型 | 🔴 |
| 2023 | [LLaVA](https://arxiv.org/abs/2304.08485) | 视觉指令微调的范式 | ⚪ |

## 七、推理模型与测试时计算

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2023 | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | 把推理变成搜索 | 🔴 |
| 2023 | [Graph of Thoughts](https://arxiv.org/abs/2308.09687) | 任意 DAG 的推理结构 | ⚪ |
| 2024 | [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | 给定预算如何分配测试时算力 | 🔴 |
| 2024 | [OpenAI o1: Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) | 长思维链的 RL 训练 | 🔴 |

## 八、评测与 Agent 基准

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2023 | [SWE-bench](https://arxiv.org/abs/2310.06770) | 真实仓库的缺陷修复评测 | 🔴 |
| 2023 | [WebArena](https://arxiv.org/abs/2307.13854) | 真实网页环境评测 | 🔴 |
| 2023 | [AgentBench](https://arxiv.org/abs/2308.03688) | 多环境 agent 评测 | ⚪ |
| 2023 | [GAIA](https://arxiv.org/abs/2311.12983) | 需要工具与多步推理的通用助手基准 | 🔴 |
| 2024 | [τ-bench](https://arxiv.org/abs/2406.12045) | 真实工具交互 + 用户模拟 | 🔴 |

## 九、综述（2026）

| 年 | 论文 | 一句话贡献 | 读 |
|---|---|---|---|
| 2026 | [Autonomous agents and multi-agent systems: a comprehensive review](https://link.springer.com/article/10.1007/s10462-026-11731-2) | 107 篇研究的统一五模块架构 | 🔴 |
| 2026 | [The evolution of tool use in LLM agents](https://link.springer.com/article/10.1007/s10462-026-11700-9) | 从单工具调用到多工具编排 | 🔴 |
| 2026 | [Architecture-level reliability in LLM-based agentic AI](https://www.oaepublish.com/articles/inect.2026.13) | 五层失效模型 | 🔴 |
