"""L12-2：位置插值（PI）与 NTK-aware 缩放。

PI 把 32k 的位置线性压到 4k 的范围内；NTK-aware 不压位置，
而是放大 base，让低频维度变"慢"，从而在不丢高频细节的前提下扩展范围。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, ax = plt.subplots(figsize=(6.8, 3.2))

m = np.linspace(0, 32768, 400)

ax.plot([0, 32768], [0, 32768], color=MUTED, lw=1.0, ls=":", label="原始位置（训练 4k 之外未见过）")
ax.plot(m, m * 4096 / 32768, color=PALETTE[0], lw=2, label="PI：线性压缩到 4k")
ax.plot(m, 4096 * (m / 4096) ** 0.55, color=PALETTE[2], lw=2, label="NTK-aware：非线性映射")

ax.axhline(4096, color=PALETTE[1], lw=0.8, ls="--")
ax.text(1200, 4300, "训练时见过的最大位置", fontsize=7, color=PALETTE[1])

ax.set_xlabel("真实位置", fontsize=8)
ax.set_ylabel("送进 RoPE 的位置", fontsize=8)
ax.set_title("两种外推思路：压位置 vs 改频率", loc="left", fontsize=9.5)
ax.legend(loc="lower right", fontsize=6.8)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "PI 会把 32k 内所有位置都压进 4k，分辨率整体下降；NTK 保留近处分辨率",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l12-interpolation")
