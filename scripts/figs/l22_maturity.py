"""L22-1：四条前沿路线的成熟度与影响。

横轴是成熟度（研究 → 原型 → 可用 → 生产），纵轴是潜在影响。
越靠右上，越值得现在投入。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

items = [
    ("线性注意力/SSM", 1.6, 4.2, PALETTE[0]),
    ("世界模型", 0.9, 4.6, PALETTE[3]),
    ("具身智能", 0.7, 4.0, PALETTE[1]),
    ("持续学习", 1.2, 3.6, PALETTE[2]),
    ("长上下文", 3.2, 3.0, PALETTE[5]),
    ("多模态统一", 2.6, 3.8, PALETTE[4]),
    ("推理模型", 2.9, 4.4, PALETTE[2]),
]

fig, ax = plt.subplots(figsize=(6.6, 3.4))
for name, x, y, color in items:
    ax.scatter([x], [y], s=90, color=color, alpha=0.75, zorder=5)
    ax.annotate(name, xy=(x, y), xytext=(x + 0.12, y + 0.12), fontsize=7, color=color)

ax.axvline(2.0, color=MUTED, lw=0.8, ls=":")
ax.axhline(3.5, color=MUTED, lw=0.8, ls=":")
ax.text(0.15, 4.85, "高影响", fontsize=6.6, color=MUTED)
ax.text(3.35, 2.55, "已落地", fontsize=6.6, color=MUTED)

ax.set_xlim(0, 4)
ax.set_ylim(2.4, 5.0)
ax.set_xticks([0.5, 1.5, 2.5, 3.5], ["研究", "原型", "可用", "生产"], fontsize=7)
ax.set_yticks([])
ax.set_xlabel("技术成熟度", fontsize=8)
ax.set_ylabel("潜在影响", fontsize=8)
ax.set_title("哪些值得现在投入", loc="left", fontsize=9.5)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "右上角（推理模型、多模态统一）已经能产生直接收益；左上角还要等",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l22-maturity")
