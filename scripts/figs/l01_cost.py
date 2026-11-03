r"""L1-4：训练成本与推理成本谁更大。

训练一次约 $6ND$ FLOPs；服务 $T$ 个 token 约 $2NT$ FLOPs。
按 Chinchilla 的 $D=20N$ 估算，两条曲线在 $N\approx17$B 处交叉——
超过这个规模，**服务成本就会超过训练成本**。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

N = np.logspace(8, 12, 300)
T = 1e12  # 累计服务 1T token

train = 6 * N * 20 * N  # 6ND，D = 20N
infer = 2 * N * T  # 2NT

cross = 2 * T / 120  # 120N² = 2NT  =>  N = 2T/120

fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(N, train, color=PALETTE[0], lw=2, label="训练一次（$6ND$，$D=20N$）")
ax.plot(N, infer, color=PALETTE[1], lw=2, label="服务 $10^{12}$ token（$2NT$）")

ax.axvline(cross, color=MUTED, lw=0.8, ls=":")
ax.scatter([cross], [6 * cross * 20 * cross], s=28, color=PALETTE[3], zorder=5)
ax.annotate(
    f"交叉点 ≈ {cross / 1e9:.0f}B",
    xy=(cross, 6 * cross * 20 * cross),
    xytext=(cross * 2.2, 6 * cross * 20 * cross * 0.25),
    fontsize=7.4,
    color=PALETTE[3],
    arrowprops=dict(arrowstyle="->", color=PALETTE[3], lw=0.8),
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("模型规模 $N$", fontsize=8)
ax.set_ylabel("FLOPs", fontsize=8)
ax.set_title("大模型的主要成本在「用」而不是「训」", loc="left", fontsize=9.5)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "这就是为什么推理优化（第 13 讲）在工程上比训练更值钱",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l01-cost")
