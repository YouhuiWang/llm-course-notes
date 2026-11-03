"""L10-1：对齐税——安全与有用的 Pareto 前沿。

把"安全权重"调高，拦截率上升，但正常请求的通过率下降。
两者不可兼得，只能选一个工作点。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

# 用一个单调曲线近似 Pareto 前沿
block = np.linspace(0.60, 0.995, 200)
useful = 0.985 - 0.42 * (block - 0.60) ** 2.1

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(block, useful, color=PALETTE[0], lw=2.2, label="可行前沿")

pts = [(0.78, "宽松\nβ=0.1"), (0.90, "平衡\nβ=0.5"), (0.97, "严格\nβ=2")]
for b, label in pts:
    u = 0.985 - 0.42 * (b - 0.60) ** 2.1
    ax.scatter([b], [u], s=36, color=PALETTE[1], zorder=5)
    ax.annotate(
        label, xy=(b, u), xytext=(b - 0.02, u + 0.035), fontsize=7, color=PALETTE[1], ha="center"
    )

ax.annotate(
    "前沿之外不可达\n前沿之内是浪费",
    xy=(0.70, 0.72),
    xytext=(0.62, 0.55),
    fontsize=7,
    color=MUTED,
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
)

ax.set_xlabel("攻击拦截率（安全）", fontsize=8)
ax.set_ylabel("正常请求通过率（有用）", fontsize=8)
ax.set_title("对齐税：安全与有用只能取其一", loc="left", fontsize=9.5)
ax.set_xlim(0.55, 1.02)
ax.set_ylim(0.5, 1.02)
ax.legend(loc="lower left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "工作点由业务决定：医疗客服要严，创意写作要松",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l10-alignment-tax")
