"""L25-1：四家模型的定位。

横轴是开放程度（闭源 → 开源），纵轴是推理与 agent 能力。
位置是定性判断，用来说明"各自押注在哪"。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

models = [
    ("GPT-5", 0.6, 4.5, PALETTE[0]),
    ("Claude 4.5", 0.7, 4.6, PALETTE[3]),
    ("Gemini 3", 0.5, 4.3, PALETTE[2]),
    ("DeepSeek-V3.2", 3.4, 4.2, PALETTE[1]),
    ("Qwen3.5", 3.6, 4.0, PALETTE[5]),
    ("Kimi K2.5", 3.5, 4.1, PALETTE[4]),
    ("Llama 4", 3.2, 3.6, MUTED),
]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
for name, x, y, color in models:
    ax.scatter([x], [y], s=80, color=color, alpha=0.8, zorder=5)
    ax.annotate(name, xy=(x, y), xytext=(x + 0.08, y + 0.1), fontsize=7, color=color)

ax.axvline(2.0, color=MUTED, lw=0.8, ls=":")
ax.text(0.15, 3.35, "闭源 API", fontsize=6.8, color=MUTED)
ax.text(2.2, 3.35, "开放权重", fontsize=6.8, color=MUTED)

ax.set_xlim(0, 4.2)
ax.set_ylim(3.2, 5.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("开放程度 →", fontsize=8)
ax.set_ylabel("推理与 agent 能力 →", fontsize=8)
ax.set_title("两条路线，同一目标", loc="left", fontsize=9.5)
clean_axes(ax, grid_axis=None)

fig.text(0.5, -0.03, "开源阵营的追赶速度，是 2023 年以来最大的变量",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l25-landscape")
