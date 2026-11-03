# 第 13 讲　推理系统

!!! info "状态"
    本讲正在撰写中（计划于 M2–M6 阶段完成）。可以先读已完成的两篇样章：[第 4 讲 注意力机制](../02-transformer/l04-attention.md)、[第 17 讲 Agent](../05-应用层/l17-agent-1.md)。

## 本讲大纲（草案）

- 显存账本：权重 / 激活 / KV cache 逐项估算
- PagedAttention 与 continuous batching
- FlashAttention 的 IO 复杂度与在线 softmax
- 量化：GPTQ、AWQ、GGUF
- 投机解码的期望加速比

## 与其他讲的关系

本讲属于课程地图的第 12 讲，前后衔接见[首页](../index.md)。

!!! note "撰写约定"
    每讲统一采用八段式：学习目标 → 直觉 → 形式化 → 推导 → 图解 → 代码 → 工程坑 → 论文 → 习题 → 速查卡。
