"""L20-4：思维树搜索。

每一步展开若干候选，用价值函数打分，保留最好的几个继续。
与贪心 CoT 的差别：允许回溯，代价是更多前向。
"""

from __future__ import annotations

from matplotlib.patches import Circle

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.4)

levels = [
    (1, [(5.0, "起点")]),
    (3, [(2.6, "想法 A"), (5.0, "想法 B"), (7.4, "想法 C")]),
    (3, [(1.6, "A1"), (3.4, "A2"), (5.0, "B1"), (6.4, "C1")]),
]

y_positions = [3.55, 2.55, 1.55]
keep = {1: [0, 1, 2], 2: [0, 1, 2, 3]}

for li, (_, nodes) in enumerate(levels):
    y = y_positions[li]
    for ni, (x, label) in enumerate(nodes):
        alive = ni in keep.get(li, [])
        color = PALETTE[0] if alive else MUTED
        ax.add_patch(Circle((x, y), 0.26, facecolor=color,
                            alpha=0.18 if alive else 0.05, edgecolor=color, lw=1.0))
        ax.text(x, y, label.split()[-1], ha="center", va="center", fontsize=5.8,
                color=color)

# 连线
ax.plot([5.0, 2.6], [3.30, 2.80], color=MUTED, lw=0.7)
ax.plot([5.0, 5.0], [3.30, 2.80], color=PALETTE[0], lw=1.2)
ax.plot([5.0, 7.4], [3.30, 2.80], color=MUTED, lw=0.7)
for x_parent, children in [(2.6, [1.6, 3.4]), (5.0, [5.0]), (7.4, [6.4])]:
    for xc in children:
        ax.plot([x_parent, xc], [2.30, 1.80], color=MUTED, lw=0.6, ls=":")

note(ax, 5.0, 0.95, "实线 = 被保留的分支；虚线 = 被剪掉的分支", MUTED, fontsize=6.8)
note(ax, 5.0, 0.55, "剪枝依据：LLM 自评（便宜但有偏）、PRM（贵但准）、环境验证（最准）",
     PALETTE[2], fontsize=6.6)
note(ax, 5.0, 0.20, "代价：每个候选都要一次前向，搜索宽度直接乘算力",
     PALETTE[1], fontsize=6.4)

save(fig, "l20-tot")
