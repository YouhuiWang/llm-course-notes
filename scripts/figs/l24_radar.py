"""L24-4：课程项目的评分雷达。

五个维度：技术深度、工程完整、评测严谨、文档清晰、创新性。
两项示例项目：一个偏工程、一个偏探索。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, save, setup

setup()

labels = ["技术深度", "工程完整", "评测严谨", "文档清晰", "创新性"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

project_a = [0.7, 0.9, 0.8, 0.85, 0.6]
project_b = [0.9, 0.6, 0.7, 0.7, 0.9]

fig, ax = plt.subplots(figsize=(5.4, 4.2), subplot_kw=dict(polar=True))

for data, name, color in [
    (project_a, "项目 A（偏工程）", PALETTE[0]),
    (project_b, "项目 B（偏探索）", PALETTE[2]),
]:
    vals = data + data[:1]
    ax.plot(angles, vals, color=color, lw=1.8, label=name)
    ax.fill(angles, vals, color=color, alpha=0.10)

ax.set_xticks(angles[:-1], labels, fontsize=7.5)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""])
ax.set_ylim(0, 1.0)
ax.set_title("评分不是单一分数，是五个维度的组合", pad=18, fontsize=9.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=7)
ax.grid(color=MUTED, lw=0.5, alpha=0.5)
ax.spines["polar"].set_color(MUTED)
ax.spines["polar"].set_linewidth(0.6)

save(fig, "l24-radar")
