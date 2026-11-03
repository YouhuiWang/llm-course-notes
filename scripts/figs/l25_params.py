"""L25-3：总参数 vs 激活参数。

稠密模型两者相等（对角线）；MoE 把总参数推到很高，
但每个 token 只算一小部分——激活占比越低，推理越便宜。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

models = [
    ("Mixtral 8×7B", 46.7, 12.9, PALETTE[2], 0.55, 0.75),
    ("Qwen3-235B", 235, 22, PALETTE[5], 0.45, 1.45),
    ("Llama 4 Maverick", 400, 17, MUTED, 0.55, 0.55),
    ("DeepSeek-V3", 671, 37, PALETTE[1], 0.35, 1.35),
    ("GLM-5", 744, 40, PALETTE[4], 0.75, 1.55),
    ("Kimi K2", 1000, 32, PALETTE[3], 0.55, 0.55),
    ("稠密 70B", 70, 70, PALETTE[0], 0.35, 1.35),
]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot([10, 1200], [10, 1200], color=MUTED, lw=0.8, ls=":", label="稠密（激活 = 总参数）")

for name, total, active, color, fx, fy in models:
    ax.scatter([total], [active], s=70, color=color, alpha=0.8, zorder=5)
    ax.annotate(f"{name}\n{active / total:.1%}", xy=(total, active),
                xytext=(total * fx, active * fy), fontsize=6.2, color=color)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 1500)
ax.set_ylim(8, 150)
ax.set_xlabel("总参数（B）", fontsize=8)
ax.set_ylabel("每 token 激活参数（B）", fontsize=8)
ax.set_title("MoE 的本质：参数很贵，计算很便宜", loc="left", fontsize=9.5)
ax.legend(loc="lower right", fontsize=7)
clean_axes(ax, grid_axis=None)

fig.text(0.5, -0.03, "激活占比越低，同样算力下能挂的「知识」越多——代价是显存",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l25-params")
