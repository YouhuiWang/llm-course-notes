# 第 5 讲　Block 与三大范式

!!! info "状态"
    本讲正在撰写中（计划于 M2–M6 阶段完成）。可以先读已完成的两篇样章：[第 4 讲 注意力机制](../02-transformer/l04-attention.md)、[第 17 讲 Agent](../05-应用层/l17-agent-1.md)。

## 本讲大纲（草案）

- Pre-LN vs Post-LN 的梯度分析
- RMSNorm 与 SwiGLU 的取舍
- 编码器 / 解码器 / 编码-解码的 mask 差异
- 预训练目标：MLM、CLM、span corruption、prefix-LM
- 参数量与 FLOPs 的逐层核算方法

## 与其他讲的关系

本讲属于课程地图的第 4 讲，前后衔接见[首页](../index.md)。

!!! note "撰写约定"
    每讲统一采用八段式：学习目标 → 直觉 → 形式化 → 推导 → 图解 → 代码 → 工程坑 → 论文 → 习题 → 速查卡。
