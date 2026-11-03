r"""L11-3：参数与计算解耦。

稠密模型每 token 的计算量正比于全部参数；
MoE 只算激活的那部分，于是同样的算力可以挂更多参数（也就是更多"记忆"）。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

N = np.logspace(9, 12, 200)

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(N, 2 * N, color=PALETTE[1], lw=2, label="稠密：FLOPs = 2N")
ax.plot(N, 2 * N / 3.6, color=PALETTE[0], lw=2, ls="--", label="MoE（top-2/8）：FLOPs ≈ 2N/3.6")

# Mixtral：总 46.7B，激活 12.9B
ax.scatter([46.7e9], [2 * 46.7e9], s=40, color=PALETTE[1], zorder=5)
ax.scatter([46.7e9], [2 * 12.9e9], s=40, color=PALETTE[0], zorder=5)
ax.annotate(
    "Mixtral 8×7B\n总 46.7B / 激活 12.9B",
    xy=(46.7e9, 2 * 12.9e9),
    xytext=(3e9, 6e11),
    fontsize=7,
    color=PALETTE[0],
    arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8),
)

ax.annotate(
    "同样的 FLOPs\n多挂 3.6 倍参数",
    xy=(2e11, 2 * 2e11 / 3.6),
    xytext=(2.2e11, 6e10),
    fontsize=7,
    color=MUTED,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("总参数量 $N$", fontsize=8)
ax.set_ylabel("每 token FLOPs", fontsize=8)
ax.set_title("MoE：显存按总参数，算力按激活参数", loc="left", fontsize=9.5)
ax.legend(loc="upper left", fontsize=7)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "代价是显存：46.7B 的权重必须全部驻留，哪怕只用 12.9B",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l11-decoupling")
