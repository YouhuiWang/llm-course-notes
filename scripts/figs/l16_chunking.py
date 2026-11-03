"""L16-2：chunk 大小的权衡。

块太小：语义不完整，召回碎片；块太大：噪声多、占 token。
实测最优点通常在 256–512 token 之间。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

size = np.array([64, 128, 256, 512, 1024, 2048])
recall = np.array([0.52, 0.66, 0.79, 0.86, 0.84, 0.78])
precision = np.array([0.88, 0.85, 0.80, 0.72, 0.61, 0.48])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(size, recall, "o-", color=PALETTE[0], lw=1.8, ms=4, label="召回率（能不能找到）")
ax.plot(size, precision, "s-", color=PALETTE[1], lw=1.8, ms=4, label="精度（找到的对不对）")

best = 512
ax.axvline(best, color=MUTED, lw=0.8, ls=":")
ax.annotate("甜点区 256–512", xy=(best, 0.86), xytext=(700, 0.90),
            fontsize=7.2, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

ax.set_xscale("log", base=2)
ax.set_xticks(size, [str(s) for s in size])
ax.set_xlabel("chunk 大小（token）", fontsize=8)
ax.set_ylabel("指标", fontsize=8)
ax.set_title("块越大召回越高，但精度越低", loc="left", fontsize=9.5)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower left", fontsize=7.2)
clean_axes(ax)

fig.text(0.5, -0.03, "重叠 10%–20% 能缓解边界切断，但会放大索引体积",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l16-chunking")
