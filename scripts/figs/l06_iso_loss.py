r"""L6-2：等损失线与等算力线的切点，就是计算最优配比。

在 $\log N$–$\log D$ 平面上：
- 等损失线（实线）是斜率为负的曲线，越靠左下损失越低；
- 等算力线（虚线）是斜率 $-1$ 的直线；
- 每条等算力线与某条等损失线相切，切点连起来就是计算最优前沿。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

N = np.logspace(8.5, 12.5, 400)
D = np.logspace(9.5, 13.5, 400)
NN, DD = np.meshgrid(N, D)

A, B, a, b = 400.0, 400.0, 0.34, 0.28
L = A / NN**a + B / DD**b + 1.6

fig, ax = plt.subplots(figsize=(6.8, 3.8))
cs = ax.contour(NN, DD, L, levels=[1.9, 2.0, 2.2, 2.6, 3.2],
                colors=[PALETTE[0]] * 5, linewidths=1.1)


# 等算力线
for c in [1e21, 1e22, 1e23, 1e24]:
    d = np.logspace(9.5, 13.5, 200)
    ax.plot(d, c / (6 * d), color=MUTED, lw=0.7, ls=":")

# 切点：N ∝ C^0.5
Cs = np.logspace(20, 25, 60)
N_opt = np.sqrt(Cs / 120)
ax.plot(N_opt, 20 * N_opt, color=PALETTE[1], lw=2, label="计算最优前沿（$D=20N$）")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e9, 3e13)
ax.set_ylim(1e10, 1e14)
ax.set_xlabel("数据量 $D$", fontsize=8)
ax.set_ylabel("参数量 $N$", fontsize=8)
ax.annotate("同一条曲线 = 同样的损失", xy=(2e11, 4e11), xytext=(3e9, 3e13),
            fontsize=7, color=PALETTE[0],
            arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8))
ax.set_title("等损失线与等算力线的切点", loc="left", fontsize=9.5)
ax.legend(loc="lower right", fontsize=7.2)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5, -0.03,
    "沿着红线走：参数与数据同比增加；偏离红线，同样的算力换不到同样的损失",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l06-iso-loss")
