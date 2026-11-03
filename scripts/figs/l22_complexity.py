"""L22-2：复杂度的分水岭。

全注意力是 O(n²)，线性注意力与 SSM 是 O(n)。
序列越长，差距越大；但线性方法的表达力也更受限。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

n = np.logspace(3, 7, 200)
d = 4096

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.loglog(n, 2 * n**2 * d, color=PALETTE[1], lw=2, label="全注意力 $O(n^2 d)$")
ax.loglog(n, 2 * n * d**2, color=PALETTE[0], lw=2, label="线性注意力 / SSM $O(n d^2)$")

cross = d
ax.axvline(cross, color=MUTED, lw=0.8, ls=":")
ax.annotate(
    f"$n=d={d}$ 处交叉",
    xy=(cross, 2 * cross * d**2),
    xytext=(2e3, 1e12),
    fontsize=7,
    color=MUTED,
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
)

ax.set_xlabel("序列长度 $n$", fontsize=8)
ax.set_ylabel("FLOPs", fontsize=8)
ax.set_title("短序列时全注意力更便宜，长序列时反过来", loc="left", fontsize=9.5)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "$n<d$ 时 $n^2d<nd^2$——这就是为什么短文本场景没必要换架构",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l22-complexity")
