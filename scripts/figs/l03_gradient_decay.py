"""L3-2：梯度沿时间步指数衰减。

BPTT 的梯度要连乘 $T$ 个雅可比矩阵。若谱半径 $\\rho<1$，梯度按 $\\rho^T$ 衰减；
$\\rho>1$ 则爆炸。只有 $\\rho\\approx1$ 时才能维持长程信号——而这一点极难调。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

T = np.arange(1, 51)
rhos = [0.90, 0.95, 0.99, 1.02]
colors = [PALETTE[1], PALETTE[3], PALETTE[0], PALETTE[2]]

fig, ax = plt.subplots(figsize=(6.6, 3.4))
for rho, c in zip(rhos, colors):
    ax.plot(T, rho**T, color=c, lw=1.8, label=rf"$\rho={rho}$")

ax.set_yscale("log")
ax.set_xlabel("回溯步数 $T$", fontsize=8)
ax.set_ylabel(r"梯度范数 $\|\partial h_T/\partial h_1\|$", fontsize=8)
ax.set_title("梯度在时间维度上指数衰减（或爆炸）", loc="left", fontsize=9.5)
ax.legend(fontsize=7.2, loc="lower left")
clean_axes(ax, grid_axis=None)

ax.annotate(
    r"$\rho=0.9$，20 步后只剩 $0.9^{20}\approx0.12$",
    xy=(20, 0.9**20),
    xytext=(21, 0.004),
    fontsize=7,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

fig.text(
    0.5, -0.03,
    "只有 ρ≈1 才留得住长程信号，但稍大就爆炸——这是 RNN 难训的根本原因",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l03-gradient-decay")
