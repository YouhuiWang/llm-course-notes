r"""L6-1：两条幂律。

固定数据量时，损失随参数量按幂律下降；固定参数量时，损失随数据量也按幂律下降。
两个指数都远小于 1，所以"翻倍规模"换来的损失下降是递减的。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

# (a) loss vs 参数量（数据固定 300B）
N = np.logspace(8, 12, 200)
loss_n = 1.7 * (N / 1e8) ** (-0.076) + 1.6
ax = axes[0]
ax.loglog(N, loss_n, color=PALETTE[0], lw=2)
ax.set_xlabel("参数量 $N$（固定 $D=300$B）", fontsize=8)
ax.set_ylabel("验证损失", fontsize=8)
ax.set_title("(a) 损失 vs 参数", loc="left", fontsize=9)
ax.annotate(
    "斜率 $-0.076$",
    xy=(3e10, 1.7 * (3e10 / 1e8) ** -0.076 + 1.6),
    xytext=(2e8, 2.55),
    fontsize=7,
    color=PALETTE[0],
    arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8),
)
clean_axes(ax, grid_axis=None)

# (b) loss vs 数据量（参数固定 1B）
D = np.logspace(9, 13, 200)
loss_d = 1.7 * (D / 1e9) ** (-0.095) + 1.6
ax = axes[1]
ax.loglog(D, loss_d, color=PALETTE[1], lw=2)
ax.set_xlabel("数据量 $D$（固定 $N=1$B）", fontsize=8)
ax.set_title("(b) 损失 vs 数据", loc="left", fontsize=9)
ax.annotate(
    "斜率 $-0.095$",
    xy=(2e11, 1.7 * (2e11 / 1e9) ** -0.095 + 1.6),
    xytext=(1e9, 2.55),
    fontsize=7,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)
clean_axes(ax, grid_axis=None)

fig.suptitle("两个指数都远小于 1：规模翻倍，收益递减", fontsize=10)
save(fig, "l06-power-laws")
