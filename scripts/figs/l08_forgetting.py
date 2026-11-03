"""L8-4：灾难性遗忘。

SFT 让目标任务快速上升，但通用能力同步下滑。
混入一部分预训练数据（虚线）能显著减缓下滑——代价是目标任务涨得慢一点。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

steps = np.linspace(0, 100, 200)

target = 0.30 + 0.62 * (1 - np.exp(-steps / 25))
general = 0.85 - 0.30 * (1 - np.exp(-steps / 40))

target_mix = 0.30 + 0.55 * (1 - np.exp(-steps / 30))
general_mix = 0.85 - 0.08 * (1 - np.exp(-steps / 45))

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(steps, target, color=PALETTE[0], lw=2, label="目标任务（纯 SFT）")
ax.plot(steps, general, color=PALETTE[1], lw=2, label="通用能力（纯 SFT）")
ax.plot(steps, target_mix, color=PALETTE[0], lw=1.6, ls="--", label="目标任务（混入预训练数据）")
ax.plot(steps, general_mix, color=PALETTE[1], lw=1.6, ls="--", label="通用能力（混入预训练数据）")

ax.annotate(
    "通用能力掉了 30 个点",
    xy=(95, 0.85 - 0.30 * (1 - np.exp(-95 / 40))),
    xytext=(45, 0.42),
    fontsize=7,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

ax.set_xlabel("SFT 步数", fontsize=8)
ax.set_ylabel("归一化得分", fontsize=8)
ax.set_title("灾难性遗忘：学会新任务，忘掉旧能力", loc="left", fontsize=9.5)
ax.set_ylim(0.25, 0.95)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.6, columnspacing=1.0)
clean_axes(ax)

fig.text(
    0.5,
    -0.34,
    "混入 5%–10% 的预训练数据，能把通用能力的下滑从 30 点压到 8 点",
    ha="center",
    fontsize=7.0,
    color=INK,
)

save(fig, "l08-forgetting")
