"""L15-2：lost in the middle。

把关键信息放在长上下文的不同位置，召回率呈 U 形：开头和结尾高，中间明显塌陷。
这与位置编码无关，是注意力分布的固有偏置。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

pos = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
recall = np.array([0.96, 0.93, 0.88, 0.80, 0.72, 0.66, 0.70, 0.78, 0.86, 0.92, 0.95])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(pos, recall, "o-", color=PALETTE[0], lw=2, ms=4)
ax.fill_between(pos, recall, 0.6, color=PALETTE[0], alpha=0.08)

ax.axhline(0.9, color=MUTED, lw=0.8, ls=":")
ax.annotate("中间塌陷：掉 30 个点", xy=(50, 0.66), xytext=(56, 0.72),
            fontsize=7.4, color=PALETTE[1],
            arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8))

ax.set_xlabel("关键信息在上下文中的相对位置（%）", fontsize=8)
ax.set_ylabel("召回率", fontsize=8)
ax.set_title("U 形曲线：中间的信息最容易被忽略", loc="left", fontsize=9.5)
ax.set_ylim(0.6, 1.0)
clean_axes(ax)

fig.text(0.5, -0.03, "对策：把关键信息放在开头或结尾，或显式重复一次",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l15-lost-middle")
