#!/usr/bin/env python3
"""生成 24 讲、8 个实验与附录的占位页。

目的：让导航立刻可用、让每讲的范围一目了然。
已被正式撰写的讲次不会被覆盖（存在即跳过）。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

LECTURES: list[tuple[str, str, str, list[str]]] = [
    ("00-导论", "l01-what-is-llm", "第 1 讲　大模型是什么", [
        "规模阈值：参数量、数据量、算力三个维度",
        "涌现能力：Wei (2022) 与 Schaeffer (2023) 的正反证据",
        "从模型到系统：LLM 应用栈的五层结构",
        "本讲义的符号约定与三条主线",
        "实验环境搭建（uv + Python 3.12）",
    ]),
    ("01-前史", "l02-statistical-lm", "第 2 讲　统计语言模型与词的表示", [
        "马尔可夫假设与 n-gram，平滑方法（Kneser-Ney）",
        "困惑度的定义与推导",
        "分布式假设与 word2vec 的 SGNS 目标函数",
        "负采样与层次 softmax 的复杂度对比",
        "局限：稀疏性、无长程依赖、无组合泛化",
    ]),
    ("01-前史", "l03-neural-seq", "第 3 讲　神经序列建模", [
        "Bengio 2003 神经语言模型",
        "RNN 与 BPTT，梯度消失的数学分析",
        "LSTM/GRU 门控机制逐项拆解",
        "seq2seq 与 Bahdanau 加性注意力",
        "核心洞察：并行性缺失为何是致命瓶颈",
    ]),
    ("02-transformer", "l05-block-and-paradigms", "第 5 讲　Block 与三大范式", [
        "Pre-LN vs Post-LN 的梯度分析",
        "RMSNorm 与 SwiGLU 的取舍",
        "编码器 / 解码器 / 编码-解码的 mask 差异",
        "预训练目标：MLM、CLM、span corruption、prefix-LM",
        "参数量与 FLOPs 的逐层核算方法",
    ]),
    ("02-transformer", "l06-scaling-laws", "第 6 讲　Scaling Laws", [
        "Kaplan 幂律与拟合方法",
        "Chinchilla 计算最优推导：$N_{opt}\\propto C^{0.5}$",
        "数据受限时代：重复训练与数据质量",
        "μP 超参迁移",
        "涌现能力的度量争议",
    ]),
    ("02-transformer", "l07-gpt-lineage", "第 7 讲　GPT 谱系与上下文学习", [
        "GPT-1/2/3 的演进路径",
        "零样本、少样本、上下文学习的机制解释",
        "三大假说：任务定位、隐式贝叶斯、梯度下降类比",
        "指令微调前夜的 prompt engineering",
        "从补全模型到对话模型的转变",
    ]),
    ("03-对齐", "l08-sft-data", "第 8 讲　指令微调与数据工程", [
        "SFT 的数据格式与 chat template",
        "数据质量 vs 数量：LIMA 与自我指导",
        "Self-Instruct / Evol-Instruct 的合成数据方法",
        "灾难性遗忘与数据混合比例",
        "评测：如何判断微调是否真的有效",
    ]),
    ("03-对齐", "l09-preference-optimization", "第 9 讲　偏好优化", [
        "Bradley-Terry 模型与奖励模型损失推导",
        "PPO 目标、KL 约束与奖励黑客",
        "DPO 的闭式推导：从 KL 正则 RL 到分类损失",
        "KTO / ORPO / SimPO / GRPO 对比",
        "RLVR：可验证奖励与 R1-Zero 现象",
    ]),
    ("03-对齐", "l10-safety-redteam", "第 10 讲　安全对齐与红队", [
        "对齐税：能力与安全的权衡",
        "拒答边界与过度拒答的度量",
        "越狱：GCG 对抗后缀的原理",
        "红队流程与自动化攻击",
        "安全评测集与治理框架",
    ]),
    ("04-架构与系统", "l11-moe", "第 11 讲　MoE 与稀疏化", [
        "稀疏激活：参数与计算解耦",
        "top-k 路由与负载均衡损失推导",
        "专家并行与通信开销",
        "DeepSeek-V3：MLA + 无辅助损失负载均衡",
        "MoE 的失败模式：路由塌缩与专家冗余",
    ]),
    ("04-架构与系统", "l12-long-context", "第 12 讲　长上下文", [
        "位置外推的困难：高频混叠",
        "位置插值 PI、NTK-aware、YaRN 推导",
        "注意力近似：滑窗、稀疏、线性注意力",
        "RULER 等长度泛化评测",
        "needle-in-haystack 的局限",
    ]),
    ("04-架构与系统", "l13-inference-systems", "第 13 讲　推理系统", [
        "显存账本：权重 / 激活 / KV cache 逐项估算",
        "PagedAttention 与 continuous batching",
        "FlashAttention 的 IO 复杂度与在线 softmax",
        "量化：GPTQ、AWQ、GGUF",
        "投机解码的期望加速比",
    ]),
    ("04-架构与系统", "l14-training-systems", "第 14 讲　训练系统", [
        "DP / TP / PP / EP 的通信量与显存权衡",
        "ZeRO 1/2/3 与 FSDP",
        "重计算、混合精度、通信重叠",
        "MFU 与吞吐估算",
        "loss spike 的诊断与恢复",
    ]),
    ("05-应用层", "l15-context-engineering", "第 15 讲　上下文工程与结构化输出", [
        "上下文窗口作为工作内存",
        "lost-in-the-middle 与位置偏置",
        "结构化输出：JSON Schema 与约束解码",
        "函数调用协议与并行调用",
        "上下文压缩：摘要、KV 压缩、前置检索",
    ]),
    ("05-应用层", "l16-rag", "第 16 讲　RAG 全链路", [
        "分块策略与嵌入模型选型（MTEB）",
        "混合检索：BM25 + 稠密向量",
        "查询改写、HyDE、路由",
        "重排与上下文压缩",
        "评估：RAGAS 四指标与失败归因",
    ]),
    ("05-应用层", "l18-agent-2", "第 18 讲　Agent（二）：协议·多智能体·可靠性", [
        "多智能体的拓扑与反直觉结论",
        "编排框架的四个共同原语",
        "MCP 生命周期、A2A 与跨组织委派",
        "Agent 训练：轨迹构造、拒绝采样、RL",
        "五层失效模型与 prompt injection 防御",
        "环境型 benchmark 与统计功效",
    ]),
    ("05-应用层", "l19-multimodal", "第 19 讲　多模态", [
        "ViT：patch 化与位置编码",
        "CLIP 对比学习目标推导",
        "视觉-语言对齐：projector 与交叉注意力",
        "扩散模型：DDPM 目标函数",
        "统一多模态架构的三种路线",
    ]),
    ("06-推理与前沿", "l20-test-time-compute", "第 20 讲　测试时计算与搜索", [
        "CoT 的机制解释与 self-consistency",
        "ToT / GoT / MCTS 搜索",
        "test-time scaling 曲线与过度思考",
        "并行采样 vs 顺序精修",
        "过程奖励模型与验证器设计",
    ]),
    ("06-推理与前沿", "l21-reasoning-training", "第 21 讲　推理模型训练", [
        "o1 / R1 范式：长 CoT 的 RL 训练",
        "RLVR 与 GRPO 目标推导",
        "冷启动、拒绝采样、蒸馏",
        "推理长度的控制与效率",
        "推理能力的评测与泛化边界",
    ]),
    ("06-推理与前沿", "l22-frontier", "第 22 讲　前沿方向", [
        "世界模型与具身智能",
        "持续学习与在线适应",
        "稀疏/线性架构的复兴（Mamba 等）",
        "AI for Science",
        "争议点清单与证据分级方法",
    ]),
    ("07-治理与实践", "l23-evaluation-governance", "第 23 讲　评测与治理", [
        "benchmark 谱系与饱和问题",
        "数据污染检测方法",
        "LLM-as-a-judge 的偏差与校准",
        "幻觉度量与安全评测",
        "合规框架与开源/闭源治理差异",
    ]),
    ("07-治理与实践", "l24-capstone", "第 24 讲　综合实践", [
        "把 24 讲串成一个系统：需求→选型→实现→评测",
        "成本、延迟、质量的三角权衡",
        "课程项目模板与评分表",
        "从原型到生产的检查清单",
        "延伸阅读路线",
    ]),
]

LABS: list[tuple[str, str, str, str]] = [
    ("lab1-attention", "Lab 1　手写注意力", "L4", "数值稳定 softmax、因果掩码、多头、RoPE；与 PyTorch 对齐到 1e-6"),
    ("lab2-minigpt", "Lab 2　迷你 GPT", "L5–L7", "2 层 4 头，TinyShakespeare，10 分钟 CPU 训练，观察 loss 曲线"),
    ("lab3-bpe", "Lab 3　手写 BPE", "L2", "训练词表、编解码、与 tiktoken 分词结果对比"),
    ("lab4-lora", "Lab 4　LoRA 微调", "L8/L15", "可训练参数占比、显存与时间对比、效果评估"),
    ("lab5-inference", "Lab 5　推理加速", "L13", "量化、KV cache、投机解码三组加速比与精度损失"),
    ("lab6-rag", "Lab 6　最小 RAG", "L16", "切分/嵌入/检索/重排/生成/评估，四指标基线 + 消融"),
    ("lab7-agent", "Lab 7　Agent 与消融", "L17", "ReAct + 三工具 + 记忆 + 轨迹可视化，四组消融"),
    ("lab8-test-time", "Lab 8　测试时计算", "L20", "采样/投票/搜索的收益曲线，复现过度思考现象"),
]


def lecture_page(title: str, bullets: list[str], n: int) -> str:
    lines = [
        f"# {title}",
        "",
        '!!! info "状态"',
        f"    本讲正在撰写中（计划于 M2–M6 阶段完成）。可以先读已完成的两篇样章："
        "[第 4 讲 注意力机制](../02-transformer/l04-attention.md)、"
        "[第 17 讲 Agent](../05-应用层/l17-agent-1.md)。",
        "",
        "## 本讲大纲（草案）",
        "",
    ]
    lines += [f"- {b}" for b in bullets]
    lines += [
        "",
        "## 与其他讲的关系",
        "",
        f"本讲属于课程地图的第 {n} 讲，前后衔接见[首页](../index.md)。",
        "",
        '!!! note "撰写约定"',
        "    每讲统一采用八段式：学习目标 → 直觉 → 形式化 → 推导 → 图解 → 代码 → 工程坑 → 论文 → 习题 → 速查卡。",
        "",
    ]
    return "\n".join(lines)


def lab_page(title: str, ref: str, desc: str) -> str:
    return "\n".join([
        f"# {title}",
        "",
        '!!! info "状态"',
        "    实验代码正在整理中。所有实验设计为 **CPU 可跑**，依赖见仓库根目录 `requirements-lab.txt`。",
        "",
        "## 目标",
        "",
        desc,
        "",
        f"## 对应讲次",
        "",
        f"[{ref}](../02-transformer/l04-attention.md) 及相关章节。",
        "",
        "## 交付四件套",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `README.md` | 目标、原理、步骤、思考题 |",
        "| `main.py` | 单文件可运行 |",
        "| `results/` | 基准输出（供对照） |",
        "| `expected.md` | 预期现象与解释 |",
        "",
    ])


def main() -> None:
    created, skipped = 0, 0
    for i, (folder, slug, title, bullets) in enumerate(LECTURES, start=1):
        path = DOCS / folder / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            skipped += 1
            continue
        path.write_text(lecture_page(title, bullets, i), encoding="utf-8")
        created += 1

    for slug, title, ref, desc in LABS:
        path = DOCS / "labs" / f"{slug}.md"
        if path.exists():
            skipped += 1
            continue
        path.write_text(lab_page(title, ref, desc), encoding="utf-8")
        created += 1

    print(f"✓ 新建 {created} 页，跳过已存在 {skipped} 页")


if __name__ == "__main__":
    main()
