"""L23-1：benchmark 的饱和。

一个基准从 40% 涨到 92%，看似进步神速，但天花板是 100%——
剩余空间越来越小，区分度越来越差，最后只能换新基准。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])
sota = np.array([0.40, 0.55, 0.68, 0.79, 0.87, 0.91, 0.93])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(years, sota, "o-", color=PALETTE[0], lw=2, ms=5)
ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
ax.text(2020.1, 0.965, "天花板 100%", fontsize=6.8, color=MUTED)

ax.fill_between(years, sota, 1.0, color=PALETTE[1], alpha=0.08)
ax.annotate(
    "剩余空间 7 个点\n区分度已经很差",
    xy=(2026, 0.965),
    xytext=(2022.2, 0.72),
    fontsize=7,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

ax.set_xlabel("年份", fontsize=8)
ax.set_ylabel("SOTA 准确率", fontsize=8)
ax.set_title("基准会饱和：越接近上限，越测不出差异", loc="left", fontsize=9.5)
ax.set_ylim(0.3, 1.02)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "饱和之后，同样的 1 个点提升，需要的模型能力差距被放大了数倍",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l23-saturation")
