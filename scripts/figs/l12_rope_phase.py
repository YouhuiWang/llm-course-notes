r"""L12-1：RoPE 的相位与波长。

第 0 维的角频率是 1 rad/位置，在 4096 的训练长度里已经转了 650 圈；
再往外推，模型见到的相位完全是没训过的区域。
低维（大波长）转得慢，负责粗定位——这就是"多尺度"的含义。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

# (a) 高频维的相位缠绕
pos = np.arange(0, 120)
ax = axes[0]
ax.plot(pos, np.mod(pos, 2 * np.pi), color=PALETTE[0], lw=1.4)
ax.set_xlabel("位置 $m$", fontsize=8)
ax.set_ylabel("相位 $(m\\theta)\\ \\mathrm{mod}\\ 2\\pi$", fontsize=8)
ax.set_title("(a) 高频维（$\\theta=1$）：每 6.28 步转一圈", loc="left", fontsize=9)
ax.annotate(
    "训练长度 4096 内已转 652 圈",
    xy=(60, 3.0),
    xytext=(8, 5.2),
    fontsize=7,
    color=PALETTE[0],
    arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8),
)
clean_axes(ax)

# (b) 各维度的波长
d = 128
i = np.arange(d // 2)
theta = 10000 ** (-2 * i / d)
ax = axes[1]
ax.semilogy(i, 2 * np.pi / theta, color=PALETTE[2], lw=1.8)
ax.axhline(4096, color=MUTED, lw=0.8, ls=":")
ax.text(2, 5200, "训练长度 4096", fontsize=7, color=MUTED)
ax.set_xlabel("维度编号 $i$", fontsize=8)
ax.set_ylabel("波长 $2\\pi/\\theta_i$", fontsize=8)
ax.set_title("(b) 波长跨越四个数量级", loc="left", fontsize=9)
ax.annotate(
    "低维：波长极长\n负责粗定位", xy=(60, 2e4), xytext=(38, 8e3), fontsize=7, color=PALETTE[2]
)
clean_axes(ax)

fig.suptitle("RoPE 的多尺度结构：高频管局部，低频管全局", fontsize=10)
save(fig, "l12-rope-phase")
