"""L1-1：参数、数据、算力三轴，以及等算力线。

$C \\approx 6ND$ 意味着在 log-log 坐标里，等算力线是一条斜率 -1 的直线。
把 GPT-3、Chinchilla、LLaMA 三个模型标上去，能直接看出"同样的算力，
参数和数据怎么分"。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, ax = plt.subplots(figsize=(6.8, 3.8))

D = np.logspace(9, 13, 200)

# 等算力线
for c, label in [
    (1e21, "$10^{21}$"),
    (1e22, "$10^{22}$"),
    (1e23, "$10^{23}$"),
    (1e24, "$10^{24}$"),
]:
    N = c / (6 * D)
    ax.plot(D, N, color=MUTED, lw=0.7, ls=":", alpha=0.8)
    # 在 D = 3e12 处标注，只标落在可视范围内的
    xlab = 3e12
    ylab = c / (6 * xlab)
    if 4e8 < ylab < 1.2e13:
        ax.text(xlab, ylab * 1.25, label, fontsize=6.8, color=MUTED, ha="center")

models = [
    ("GPT-3\n175B / 300B", 175e9, 300e9, PALETTE[1], 0.30, 2.6),
    ("Chinchilla\n70B / 1.4T", 70e9, 1.4e12, PALETTE[2], 0.28, 2.4),
    ("LLaMA-7B\n6.7B / 1T", 6.7e9, 1e12, PALETTE[0], 0.30, 0.22),
]
for name, n, d, c, fx, fy in models:
    ax.scatter([d], [n], s=40, color=c, zorder=5)
    ax.annotate(name, xy=(d, n), xytext=(d * fx, n * fy), fontsize=7, color=c)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e9, 3e13)
ax.set_ylim(4e8, 1.2e13)
ax.set_xlabel("训练数据 $D$（token）", fontsize=8)
ax.set_ylabel("参数量 $N$", fontsize=8)
ax.set_title("等算力线：$C\\approx 6ND$", loc="left", fontsize=9.5)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "同一条虚线上算力相同：往上走是「大模型少数据」，往右走是「小模型多数据」",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l01-three-axes")
