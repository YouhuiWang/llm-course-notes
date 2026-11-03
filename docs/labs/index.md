# 实验总览

> 八个实验覆盖全书的技术主线。**全部 CPU 可跑**（Apple Silicon / x86 均可），GPU 仅作可选加速。
> 每个实验固定四件套：`README.md`（目标与步骤）、`main.py`（单文件可运行）、`results/`（基准输出）、`expected.md`（预期现象与解释）。

## 环境

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lab.txt
```

| 依赖 | 用途 | 说明 |
|---|---|---|
| `torch>=2.4` | Lab 1/2/4/5 | CPU 版即可 |
| `numpy` | 全部 | — |
| `matplotlib` | 可视化 | 复用 `scripts/figs/style.py` 的配色 |
| `tiktoken` | Lab 3 对照 | 可选，缺失时跳过对比 |

## 清单

| Lab | 名称 | 对应讲次 | 核心产出 | 预估耗时 |
|---|---|---|---|---|
| [1](lab1-attention.md) | 手写注意力 | L4 | 与 PyTorch 对齐到 $10^{-6}$，RoPE 性质验证 | 30 min |
| [2](lab2-minigpt.md) | 迷你 GPT | L5–L7 | loss 曲线收敛，生成通顺短句 | 60 min |
| [3](lab3-bpe.md) | 手写 BPE | L2 | 词表训练 + 与 tiktoken 对比报告 | 40 min |
| [4](lab4-lora.md) | LoRA 微调 | L8/L15 | 可训练参数占比、显存/时间对比 | 45 min |
| [5](lab5-inference.md) | 推理加速 | L13 | 量化/KV cache/投机解码三组加速比 | 60 min |
| [6](lab6-rag.md) | 最小 RAG | L16 | 四指标基线 + 消融 | 60 min |
| [7](lab7-agent.md) | Agent 与消融 | L17 | 四组消融的成功率与成本表 | 90 min |
| [8](lab8-test-time.md) | 测试时计算 | L20 | 采样/投票/搜索的收益曲线 | 60 min |

## 实验设计原则

1. **最小可运行**：每个 `main.py` 单文件、无隐藏配置，`python main.py` 直接出结果。
2. **有对照**：所有实验都要和一个"标准实现"或"基线"对比，否则无法判断对错。
3. **可归因**：失败时要能定位到具体环节（数值、mask、形状、超参）。
4. **有思考题**：实验报告要求回答 3–5 个"如果改这里会怎样"的问题。

## 提交格式

课程使用时的建议评分表：

| 项 | 权重 | 要求 |
|---|---|---|
| 跑通 | 30% | `main.py` 一次运行成功，输出符合 `expected.md` |
| 对齐 | 25% | 与参考实现误差在阈值内 |
| 分析 | 30% | 消融表 + 失败归因 + 与讲义的对照 |
| 思考题 | 15% | 有独立见解，不是复述讲义 |
