"""L11-2：路由塌缩。

不加约束时，路由会迅速把大部分 token 送给少数专家，其余专家"饿死"。
辅助损失（load balancing loss）把负载拉回均匀。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)

experts = [f"E{i}" for i in range(1, 9)]
collapsed = np.array([0.42, 0.28, 0.14, 0.07, 0.04, 0.03, 0.01, 0.01])
balanced = np.full(8, 0.125) + np.array([0.01, -0.01, 0.005, -0.005, 0.008, -0.008, 0.004, -0.004])

for ax, data, title, color in [
    (axes[0], collapsed, "(a) 无负载均衡：塌缩", PALETTE[1]),
    (axes[1], balanced, "(b) 加辅助损失：均匀", PALETTE[0]),
]:
    ax.bar(experts, data, color=color, width=0.62)
    ax.axhline(0.125, color=MUTED, lw=0.8, ls="--")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("承接 token 的比例", fontsize=7.5)
    clean_axes(ax)

axes[0].annotate("前两个专家吃掉 70%", xy=(1, 0.28), xytext=(2.4, 0.40),
                 fontsize=7, color=PALETTE[1],
                 arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8))

fig.text(0.5, -0.04, "虚线是理想均匀值 1/8 = 12.5%", ha="center", fontsize=7.2, color=INK)

save(fig, "l11-collapse")
