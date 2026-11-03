"""L14-3：激活重计算的取舍。

不重计算：激活全存，显存爆；每层重计算：只存层边界，显存降一个数量级，
代价是多一次前向（约 +33% 算力）。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

modes = ["不重计算", "整层重计算", "子层重计算"]
mem = [8.6, 1.1, 0.6]          # GB
overhead = [0, 33, 45]          # 额外算力百分比

fig, ax = plt.subplots(figsize=(6.6, 3.2))
x = np.arange(3)
bars = ax.bar(x, mem, color=[PALETTE[1], PALETTE[0], PALETTE[2]], width=0.55)
for bar, m, o in zip(bars, mem, overhead):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.25, f"{m} GB", ha="center", fontsize=7.6)
    ax.text(bar.get_x() + bar.get_width() / 2, 0.25, f"+{o}% 算力" if o else "基准",
            ha="center", fontsize=6.8, color="white")

ax.set_xticks(x, modes)
ax.set_ylabel("激活显存（GB）", fontsize=8)
ax.set_title("重计算：用算力换显存", loc="left", fontsize=9.5)
ax.set_ylim(0, 10)
clean_axes(ax)

fig.text(0.5, -0.03, "7B 模型、序列 2048、batch 8 的量级；显存降 8 倍，算力只多三成",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l14-recompute")
