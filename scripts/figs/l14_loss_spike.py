"""L14-4：loss spike。

长跑训练里 loss 突然跳升，常见于：数据里有异常样本、学习率过高、
fp16 溢出、某个专家/层梯度爆炸。处理方式是回滚 + 跳过数据 + 降 lr。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

rng = np.random.default_rng(3)
steps = np.arange(0, 2000)
loss = 2.6 * np.exp(-steps / 420) + 1.55 + rng.normal(0, 0.012, size=steps.size)

# 注入两次尖峰
for center, height in [(760, 0.55), (1480, 0.9)]:
    loss += height * np.exp(-((steps - center) ** 2) / 60)

fig, ax = plt.subplots(figsize=(6.8, 3.2))
ax.plot(steps, loss, color=PALETTE[0], lw=0.9)
ax.axvline(760, color=PALETTE[1], lw=0.8, ls=":")
ax.axvline(1480, color=PALETTE[1], lw=0.8, ls=":")

ax.annotate(
    "小尖峰：跳过该 batch\n继续训",
    xy=(760, 2.2),
    xytext=(830, 2.55),
    fontsize=6.8,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)
ax.annotate(
    "大尖峰：回滚到\n上一个 checkpoint",
    xy=(1480, 2.55),
    xytext=(1560, 2.75),
    fontsize=6.8,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

ax.set_xlabel("训练步数", fontsize=8)
ax.set_ylabel("训练损失", fontsize=8)
ax.set_title("长跑训练的常态：偶发尖峰", loc="left", fontsize=9.5)
ax.set_ylim(1.4, 3.0)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "判断标准：能否在 100 步内自己恢复；恢复不了就回滚",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l14-loss-spike")
