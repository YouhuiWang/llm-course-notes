"""L12-3：长度外推的效果。

训练长度 4k，测试长度一路加到 32k。
不缩放时 4k 之后断崖；PI 撑得久一些；YaRN 在 32k 仍保留大部分性能。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

L = np.array([4096, 8192, 12288, 16384, 24576, 32768])
none = np.array([0.86, 0.62, 0.44, 0.31, 0.18, 0.11])
pi = np.array([0.85, 0.81, 0.76, 0.71, 0.62, 0.52])
yarn = np.array([0.86, 0.845, 0.83, 0.81, 0.77, 0.72])

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(L / 1024, none, "o-", color=PALETTE[1], lw=1.8, ms=4, label="不缩放")
ax.plot(L / 1024, pi, "s-", color=PALETTE[0], lw=1.8, ms=4, label="位置插值 PI")
ax.plot(L / 1024, yarn, "^-", color=PALETTE[2], lw=1.8, ms=4, label="YaRN")

ax.axvline(4, color=MUTED, lw=0.8, ls=":")
ax.text(4.3, 0.20, "训练长度", fontsize=7, color=MUTED, rotation=90)
ax.annotate("超出训练长度后\n断崖式下跌", xy=(8, 0.62), xytext=(11, 0.30),
            fontsize=7, color=PALETTE[1],
            arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8))

ax.set_xlabel("测试长度（k token）", fontsize=8)
ax.set_ylabel("长文本任务得分", fontsize=8)
ax.set_title("同样训练 4k，外推能力差别很大", loc="left", fontsize=9.5)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=7.2)
clean_axes(ax)

fig.text(0.5, -0.03, "数据为示意量级：真实曲线的形状与任务强相关",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l12-extrapolation")
