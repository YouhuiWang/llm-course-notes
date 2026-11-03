"""L9-2：Bradley-Terry 模型把"分数差"映射成"偏好概率"。

横轴是两条回答的奖励差，纵轴是前者被偏好的概率。
标出例 9.1 的两个点，说明"差 1 分"只对应 73% 的把握。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

delta = np.linspace(-6, 6, 400)
p = 1 / (1 + np.exp(-delta))

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(delta, p, color=PALETTE[0], lw=2, label=r"$\sigma(\Delta r)$")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.axvline(0, color=MUTED, lw=0.8, ls=":")

pts = [(1.0, 0.731), (3.0, 0.953)]
for dx, py in pts:
    ax.scatter([dx], [py], s=30, color=PALETTE[1], zorder=5)
    ax.annotate(
        f"$\\Delta r={dx:.0f}$ → {py:.2f}",
        xy=(dx, py),
        xytext=(dx + 0.35, py - 0.16),
        fontsize=7.4,
        color=PALETTE[1],
    )

ax.set_xlabel(r"奖励差 $\Delta r = r(x,y_w) - r(x,y_l)$", fontsize=8)
ax.set_ylabel("$y_w$ 被偏好的概率", fontsize=8)
ax.set_title("偏好概率只看「差」，不看绝对值", loc="left", fontsize=9.5)
ax.set_ylim(0, 1.02)
ax.legend(loc="upper left", fontsize=7.5)
clean_axes(ax)

fig.text(
    0.5, -0.02,
    "差 1 分只有 73% 的把握；要 95% 的把握，差需要到 3 分",
    ha="center", fontsize=7.4, color=INK,
)

save(fig, "l09-bt-curve")
