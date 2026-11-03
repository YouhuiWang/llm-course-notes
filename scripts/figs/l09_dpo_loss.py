"""L9-4：DPO 的损失与梯度权重。

(a) 损失随"对数比之差"的变化：β 越大，曲线越陡。
(b) 梯度权重 σ(-βΔ)：已经分得很开的样本权重趋近 0，
    这就是 DPO 自动"不再管已学会的样本"的机制。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

delta = np.linspace(-4, 4, 400)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

ax = axes[0]
for b, c in [(0.1, PALETTE[0]), (1.0, PALETTE[1])]:
    ax.plot(delta, -np.log(sigmoid(b * delta)), color=c, lw=1.8, label=rf"$\beta={b}$")
ax.set_xlabel(r"$\Delta$（对数比之差）", fontsize=8)
ax.set_ylabel("DPO 损失", fontsize=8)
ax.set_title("(a) 损失：β 越大越陡", loc="left", fontsize=9)
ax.set_ylim(0, 2.2)
ax.legend(fontsize=7.4)
clean_axes(ax)

ax = axes[1]
for b, c in [(0.1, PALETTE[0]), (1.0, PALETTE[1])]:
    ax.plot(delta, sigmoid(-b * delta), color=c, lw=1.8, label=rf"$\beta={b}$")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel(r"$\Delta$", fontsize=8)
ax.set_ylabel("梯度权重 $\\sigma(-\\beta\\Delta)$", fontsize=8)
ax.set_title("(b) 梯度权重：学好的样本自动降权", loc="left", fontsize=9)
ax.legend(fontsize=7.4)
clean_axes(ax)

fig.text(
    0.5, -0.04,
    "Δ 越大（模型已经更偏好 $y_w$），权重越小 —— DPO 自带「难度感知」",
    ha="center", fontsize=7.4, color=INK,
)

save(fig, "l09-dpo-loss")
