"""L1-2：所谓"涌现"，很大程度上取决于你用什么指标去量。

同一个模型能力，用"全对才算对"（exact match）去量，曲线像阶跃；
用"token 级准确率"去量，是同一条平滑的 S 曲线。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

x = np.logspace(8, 12, 200)
logx = np.log10(x)

# 潜在能力：平滑上升
skill = 1 / (1 + np.exp(-(logx - 10.2) * 2.2))

# 指标一：token 级准确率（平滑）
token_acc = 0.02 + 0.94 * skill

# 指标二：整题全对（能力要接近 1 才可能全对，n=8 步）
exact = token_acc**8

fig, ax = plt.subplots(figsize=(6.8, 3.5))
ax.plot(x, token_acc, color=PALETTE[0], lw=2, label="token 级准确率（平滑）")
ax.plot(x, exact, color=PALETTE[1], lw=2, label="整题全对（看似阶跃）")

ax.axvspan(3e10, 1e11, color=PALETTE[1], alpha=0.07)
ax.annotate(
    "同一段能力，\n换指标就成了「涌现」",
    xy=(6e10, 0.35),
    xytext=(2.2e9, 0.62),
    fontsize=7.4,
    color=INK,
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
)

ax.set_xscale("log")
ax.set_xlabel("模型规模（参数量）", fontsize=8)
ax.set_ylabel("指标得分", fontsize=8)
ax.set_title("「涌现」可能是度量方式的产物", loc="left", fontsize=9.5)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "换成线性坐标、换更细的刻度、换更平滑的指标，阶跃往往会消失",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l01-emergence")
