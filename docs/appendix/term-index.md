# 术语索引（按讲次）

> 本页由 `scripts/gen_term_index.py` 从[术语表](glossary.md)生成，请勿手工编辑。
> 用途：按学习进度回查——读到第 N 讲时，本讲引入了哪些新词。

共 **113** 条术语，分布在 23 个讲次。

## 第 1 讲

| 术语 | English | 定义 |
|---|---|---|
| Top-p 采样 | Nucleus Sampling | 只在累积概率前 p 的候选集内采样 |
| 上下文窗口 | Context Window | 单次前向能处理的最大 token 数，即模型的"工作内存" |
| 令牌 | Token | 模型处理的最小文本单位，由分词器切分 |
| 涌现能力 | Emergent Ability | 规模超过阈值后突然出现的能力（存在度量争议） |
| 温度 | Temperature | 采样时对 logits 的缩放系数，控制随机性 |

## 第 2 讲

| 术语 | English | 定义 |
|---|---|---|
| 分布式假设 | Distributional Hypothesis | 上下文相似的词，语义也相似 |
| 困惑度 | Perplexity | 平均每 token 的负对数似然的指数，语言模型的标准指标 |
| 字节对编码 | BPE | 从字符出发不断合并高频相邻对，得到子词词表 |
| 词嵌入 | Word Embedding | 把离散词映射到连续向量的稠密表示 |

## 第 3 讲

| 术语 | English | 定义 |
|---|---|---|
| 注意力机制 | Attention | 用查询与键的相似度对值做加权求和，使模型能按需取用上下文信息 |

## 第 4 讲

| 术语 | English | 定义 |
|---|---|---|
| 位置编码 | Positional Encoding | 给注意力注入顺序信息，打破置换等变性 |
| 分组查询注意力 | GQA | 多个查询头共享一组键值头，降低 KV cache 显存 |
| 因果掩码 | Causal Mask | 上三角置为负无穷，阻止位置 $t$ 看到 $t$ 之后的信息 |
| 多头注意力 | Multi-Head Attention | 在多个低维子空间并行做注意力，再拼接投影 |
| 旋转位置编码 | RoPE | 用复数旋转把绝对位置写入相位，使内积只依赖相对距离 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 点积相似度除以 $\sqrt{d_k}$ 后过 softmax 的注意力形式 |
| 自注意力 | Self-Attention | 查询、键、值都来自同一序列的注意力 |

## 第 5 讲

| 术语 | English | 定义 |
|---|---|---|
| 前馈网络 | FFN | Transformer block 中的逐位置两层 MLP，可视为键值记忆 |
| 因果语言建模 | CLM | 用前文预测下一个 token，GPT 系列的预训练目标 |
| 层归一化 | LayerNorm | 对每个样本的特征维做归一化，稳定深层训练 |
| 掩码语言建模 | MLM | 随机遮住部分 token 让模型还原，BERT 的预训练目标 |
| 残差连接 | Residual Connection | 把子层输出与输入相加，保证梯度有直通路径 |
| 编码器 | Encoder | 双向可见全序列的 Transformer 堆叠，适合理解类任务 |
| 解码器 | Decoder | 带因果掩码、逐 token 生成的 Transformer 堆叠 |
| 预归一化 | Pre-LN | 把归一化放在子层输入侧，是深层 Transformer 稳定的关键 |

## 第 6 讲

| 术语 | English | 定义 |
|---|---|---|
| 缩放定律 | Scaling Law | 损失随参数量、数据量、算力呈幂律下降的经验规律 |
| 计算最优 | Compute-Optimal | 给定算力时让损失最小的参数/数据配比（Chinchilla） |

## 第 7 讲

| 术语 | English | 定义 |
|---|---|---|
| 上下文学习 | In-Context Learning | 不改参数，仅靠提示中的示例完成任务 |
| 少样本提示 | Few-shot Prompting | 在提示中给少量示例引导模型输出格式 |

## 第 8 讲

| 术语 | English | 定义 |
|---|---|---|
| 指令微调 | Instruction Tuning | 用"指令–回答"对做监督微调，让模型学会遵循指令 |
| 灾难性遗忘 | Catastrophic Forgetting | 微调新任务时旧能力显著退化 |
| 监督微调 | SFT | 在标注数据上做标准交叉熵训练，对齐流程的第一步 |

## 第 9 讲

| 术语 | English | 定义 |
|---|---|---|
| 人类反馈强化学习 | RLHF | 用奖励模型 + PPO 优化策略，使其输出更符合偏好 |
| 奖励模型 | Reward Model | 学习人类偏好打分函数，替代人工给 RL 提供奖励 |
| 直接偏好优化 | DPO | 把 KL 正则的 RL 目标解析地转成分类损失，免去 RL 循环 |
| 群体相对策略优化 | GRPO | 用组内相对优势替代价值网络，降低 RL 显存开销 |

## 第 10 讲

| 术语 | English | 定义 |
|---|---|---|
| 对抗后缀 | Adversarial Suffix | 用梯度搜出的一串字符，诱导模型违规 |
| 对齐税 | Alignment Tax | 对齐训练带来的通用能力下降 |
| 对齐税 | Alignment Tax | 安全对齐带来的通用能力下降 |
| 红队测试 | Red Teaming | 主动构造攻击以发现模型的安全边界 |
| 越狱 | Jailbreak | 通过构造输入绕过模型的安全拒答 |
| 越狱 | Jailbreak | 构造输入绕过模型的安全拒答 |
| 间接注入 | Indirect Prompt Injection | 把恶意指令藏在检索内容或工具返回里 |

## 第 11 讲

| 术语 | English | 定义 |
|---|---|---|
| 专家并行 | Expert Parallelism | 把不同专家放到不同设备上的并行策略 |
| 容量因子 | Capacity Factor | 每个专家最多接收的 token 数上限 |
| 混合专家 | MoE | 每个 token 只激活少数专家，用稀疏性换容量 |
| 稀疏激活 | Sparse Activation | 每个 token 只计算少数专家的参数 |
| 细粒度专家 | Fine-grained Experts | 把专家切得更小更多，组合更灵活 |
| 负载均衡损失 | Load Balancing Loss | 防止路由塌缩到少数专家的辅助损失 |

## 第 12 讲

| 术语 | English | 定义 |
|---|---|---|
| 位置插值 | Position Interpolation | 把长序列位置压缩回训练范围以支持外推 |
| 分频段外推 | YaRN | 高频不缩放、低频插值、加温度修正的外推方法 |
| 长上下文外推 | Context Extrapolation | 在训练长度之外仍保持性能的能力 |

## 第 13 讲

| 术语 | English | 定义 |
|---|---|---|
| 分页注意力 | PagedAttention | 用分页表管理 KV cache，消除显存碎片 |
| 分页注意力 | PagedAttention | 把 KV cache 切成固定块按需分配 |
| 投机解码 | Speculative Decoding | 小模型草稿 + 大模型校验，无损加速生成 |
| 显存带宽瓶颈 | Memory-bound | 推理时瓶颈在读权重而非算力 |
| 蒸馏 | Distillation | 用大模型输出训练小模型，迁移能力 |
| 连续批处理 | Continuous Batching | 请求一结束就补新请求，提升槽位利用率 |
| 量化 | Quantization | 用低比特表示权重/激活，换取显存与带宽 |
| 键值缓存 | KV Cache | 缓存历史 token 的 K/V，避免推理时重复计算 |

## 第 14 讲

| 术语 | English | 定义 |
|---|---|---|
| 张量并行 | Tensor Parallelism | 把单层矩阵切到多卡，层内通信 |
| 数据分片 | Sharding | 把参数/梯度/优化器状态切到多卡 |
| 数据并行 | Data Parallelism | 每卡一份完整模型，切分数据、同步梯度 |
| 模型浮点利用率 | MFU | 实际算力占硬件峰值的比例，衡量训练效率 |
| 流水并行 | Pipeline Parallelism | 按层切分模型，流水线式执行 |
| 混合精度 | Mixed Precision | 用 bf16/fp16 计算、fp32 累加，兼顾速度与稳定 |
| 重计算 | Activation Checkpointing | 前向不存激活，反向重算以省显存 |
| 零冗余优化 | ZeRO | 三阶段切分优化器状态、梯度与参数 |

## 第 15 讲

| 术语 | English | 定义 |
|---|---|---|
| 上下文工程 | Context Engineering | 把有限窗口组织成最有效的工作内存 |
| 中间遗忘 | Lost in the Middle | 长上下文中间位置的信息召回率最低 |
| 低秩适配 | LoRA | 用两个低秩矩阵的乘积表示权重增量 |
| 前缀缓存 | Prefix Caching | 复用固定前缀的 KV，省重复计算 |
| 参数高效微调 | PEFT | 只训练少量附加参数（如 LoRA）适配下游任务 |
| 约束解码 | Constrained Decoding | 用有限状态机在每步屏蔽非法 token |
| 结构化输出 | Structured Output | 约束解码使输出严格符合 JSON/Schema |

## 第 16 讲

| 术语 | English | 定义 |
|---|---|---|
| 倒数排名融合 | RRF | 只用排名融合多路检索结果，无需归一化 |
| 假设文档嵌入 | HyDE | 先让模型写假答案，再用它检索 |
| 分块 | Chunking | 把长文档切成适合检索的语义单元 |
| 忠实度 | Faithfulness | 答案中每句话有上下文支撑的比例 |
| 检索增强生成 | RAG | 先检索外部知识再生成，缓解幻觉与知识过期 |
| 混合检索 | Hybrid Retrieval | 融合 BM25 稀疏检索与向量稠密检索 |
| 重排 | Reranking | 用交叉编码器对召回结果精排 |

## 第 17 讲

| 术语 | English | 定义 |
|---|---|---|
| 反应式循环 | ReAct | 推理（Thought）与行动（Action）交替的 agent 范式 |
| 反思 | Reflection | 失败后总结教训并改变策略 |
| 工具调用 | Tool Use / Function Calling | 模型输出结构化调用，由外部执行并回填结果 |
| 智能体 | Agent | 由 LLM 充当策略、能调用工具并在环境中多步行动的系統 |
| 误差累积 | Error Accumulation | 多步任务的整体成功率按 $p^n$ 衰减 |

## 第 18 讲

| 术语 | English | 定义 |
|---|---|---|
| 多智能体系统 | Multi-Agent System | 多个 agent 通过消息协作完成任务 |
| 提示注入 | Prompt Injection | 把恶意指令藏进数据，劫持 agent 行为 |
| 智能体间协议 | A2A | 让不同框架的 agent 互相发现与委派 |
| 模型上下文协议 | MCP | 统一 LLM 与外部工具/资源连接的开放协议 |

## 第 19 讲

| 术语 | English | 定义 |
|---|---|---|
| 对比语言图像预训练 | CLIP | 用图文配对做对比学习，得到共享嵌入空间 |
| 扩散模型 | Diffusion Model | 学习逐步去噪，从随机噪声生成数据 |
| 投影层 | Projector | 把视觉特征映射到语言模型词向量空间 |
| 视觉 Transformer | ViT | 把图像切 patch 后当 token 序列处理 |
| 视觉指令微调 | Visual Instruction Tuning | 用图文对话数据训练多模态助手 |

## 第 20 讲

| 术语 | English | 定义 |
|---|---|---|
| 思维链 | Chain-of-Thought | 让模型显式写出中间推理步骤 |
| 测试时计算 | Test-Time Compute | 推理阶段多花算力（采样/搜索/反思）换准确率 |
| 自一致性 | Self-Consistency | 多次采样后取多数答案 |
| 过度思考 | Overthinking | 思考过长导致准确率反而下降 |
| 过程奖励模型 | PRM | 对推理的每一步打分，而非只评最终答案 |

## 第 21 讲

| 术语 | English | 定义 |
|---|---|---|
| 可验证奖励 | RLVR | 用规则或测试用例判对错，为推理任务提供可靠奖励 |
| 可验证奖励 | RLVR | 用规则或测试判定对错，替代人类偏好 |
| 拒绝采样 | Rejection Sampling | 同题多次采样，只保留成功轨迹 |
| 推理模型 | Reasoning Model | 经 RL 训练、会生成长思维链再作答的模型 |
| 组内相对优势 | Group Relative Advantage | GRPO 用组内均值作基线算优势 |

## 第 22 讲

| 术语 | English | 定义 |
|---|---|---|
| 世界模型 | World Model | 学习环境动态、可预测行动后果的内部模型 |
| 持续学习 | Continual Learning | 顺序学新任务而不遗忘旧任务 |
| 状态空间模型 | SSM | 用线性递推代替注意力，复杂度 $O(n)$ |

## 第 23 讲

| 术语 | English | 定义 |
|---|---|---|
| 基准测试 | Benchmark | 标准化评测集，用于横向比较模型能力 |
| 大模型即裁判 | LLM-as-a-Judge | 用强模型给输出打分，需校准位置/长度偏差 |
| 幻觉 | Hallucination | 模型生成看似合理但与事实不符的内容 |
| 数据污染 | Data Contamination | 评测题泄漏进训练集，导致分数虚高 |
