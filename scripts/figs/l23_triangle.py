"""L23-4：评测的三角约束。

有效性（测的是不是真能力）、可靠性（重复测是否一致）、成本（人力与金钱）。
三者只能取其二——这是所有评测设计的根本矛盾。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

fig, ax = plt.subplots(figsize=(5.6, 3.6))

angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 3, endpoint=False)
labels = ["有效性\n（测真本事）", "可靠性\n（重复一致）", "成本\n（人力/金钱）"]
colors = [PALETTE[0], PALETTE[2], PALETTE[1]]

for ang, lab, c in zip(angles, labels, colors):
    x, y = np.cos(ang), np.sin(ang)
    ax.scatter([x], [y], s=140, color=c, alpha=0.75, zorder=5)
    ax.annotate(lab, xy=(x, y), xytext=(x * 1.35, y * 1.35), fontsize=7.2,
                color=c, ha="center", va="center")

ax.plot([np.cos(angles[0]), np.cos(angles[1])], [np.sin(angles[0]), np.sin(angles[1])],
        color=MUTED, lw=0.8)
ax.plot([np.cos(angles[1]), np.cos(angles[2])], [np.sin(angles[1]), np.sin(angles[2])],
        color=MUTED, lw=0.8)
ax.plot([np.cos(angles[2]), np.cos(angles[0])], [np.sin(angles[2]), np.sin(angles[0])],
        color=MUTED, lw=0.8)

ax.text(0, 0.15, "选两个\n放弃第三个", ha="center", fontsize=8, color=INK)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.5, 1.6)
ax.axis("off")

fig.text(0.5, -0.02, "想三者兼得，通常意味着某一项被偷偷放松了",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l23-triangle")
