"""L20-3：多数投票的收益。

单次正确率 p 越高，投票越有用；k 增大时边际收益递减。
p 接近 0.5 时投票几乎没用——因为错误答案也会被投出来。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from math import comb

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

ks = np.arange(1, 32, 2)


def majority(p: float, k: int) -> float:
    need = k // 2 + 1
    return sum(comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(need, k + 1))


fig, ax = plt.subplots(figsize=(6.6, 3.2))
for p, color in [(0.4, PALETTE[1]), (0.6, PALETTE[0]), (0.8, PALETTE[2])]:
    ax.plot(ks, [majority(p, int(k)) for k in ks], "o-", color=color, lw=1.8, ms=3.5,
            label=f"单次 $p={p}$")

ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("采样次数 $k$", fontsize=8)
ax.set_ylabel("多数投票后的正确率", fontsize=8)
ax.set_title("投票能放大优势，也能放大劣势", loc="left", fontsize=9.5)
ax.set_ylim(0, 1.02)
ax.legend(loc="center right", fontsize=7.2)
clean_axes(ax)

fig.text(0.5, -0.03, "p=0.4 时投票反而更差——错答案占多数",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l20-voting")
