"""L16-3：稀疏 vs 稠密 vs 混合。

BM25 擅长关键词精确匹配，向量检索擅长语义近似，混合检索在两类上都接近最优。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

types = ["关键词型查询\n（型号、编号）", "语义型查询\n（换个说法）", "混合型查询\n（真实场景）"]
bm25 = [0.92, 0.48, 0.70]
dense = [0.55, 0.88, 0.72]
hybrid = [0.91, 0.87, 0.86]

x = np.arange(3)
w = 0.26

fig, ax = plt.subplots(figsize=(6.8, 3.3))
ax.bar(x - w, bm25, w, color=PALETTE[1], label="BM25（稀疏）")
ax.bar(x, dense, w, color=PALETTE[0], label="向量检索（稠密）")
ax.bar(x + w, hybrid, w, color=PALETTE[2], label="混合 + RRF")

for i in range(3):
    ax.text(x[i] - w, bm25[i] + 0.015, f"{bm25[i]:.2f}", ha="center", fontsize=6.4)
    ax.text(x[i], dense[i] + 0.015, f"{dense[i]:.2f}", ha="center", fontsize=6.4)
    ax.text(x[i] + w, hybrid[i] + 0.015, f"{hybrid[i]:.2f}", ha="center", fontsize=6.4)

ax.set_xticks(x, types, fontsize=7)
ax.set_ylabel("Recall@10", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("没有一种检索能通吃", loc="left", fontsize=9.5)
ax.legend(loc="lower right", fontsize=7)
clean_axes(ax)

fig.text(0.5, -0.05, "混合检索用 RRF 融合两路排名，代价是多一次检索与融合计算",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l16-hybrid")
