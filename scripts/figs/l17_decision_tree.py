"""L17-1：Agent 还是 Workflow 的判定树。

三个问题，四条出口。核心判据是"谁决定控制流"。
"""

from __future__ import annotations

from style import PALETTE, MUTED, save, setup
from diagram import arrow, box, canvas, note, tag

setup()

fig, ax = canvas(6.8, 3.5, xmax=10, ymax=5.2)

QW, QH = 3.1, 0.72
qx = 3.5

box(ax, qx, 4.05, QW, QH, "步骤是否预先完全已知？", PALETTE[0], fontsize=7.8)
box(ax, qx, 2.85, QW, QH, "需要按中间结果动态分支？", PALETTE[0], fontsize=7.8)
box(ax, qx, 1.65, QW, QH, "错误代价高且不可逆？", PALETTE[0], fontsize=7.8)

# 出口
box(ax, 0.25, 4.05, 2.55, QH, "Workflow\n代码写死流程", PALETTE[2], fontsize=7.6)
box(ax, 0.25, 2.85, 2.55, QH, "Workflow\n（判据：无需动态决策）", PALETTE[2], fontsize=7.2)
box(ax, 7.15, 1.65, 2.6, QH, "Agent\n+ 人工审批门", PALETTE[1], fontsize=7.6)
box(ax, qx, 0.35, QW, QH, "Agent（ReAct）\n最简单的自由循环", PALETTE[3], fontsize=7.4)

# 箭头
arrow(ax, (qx, 4.41), (2.8, 4.41), color=PALETTE[2])
tag(ax, 3.15, 4.60, "是", PALETTE[2])

arrow(ax, (qx, 3.21), (2.8, 3.21), color=PALETTE[2])
tag(ax, 3.15, 3.40, "否", PALETTE[2])

arrow(ax, (qx + QW / 2, 4.05), (qx + QW / 2, 3.57), color=MUTED)
tag(ax, qx + QW / 2 + 0.28, 3.81, "否", MUTED)

arrow(ax, (qx + QW / 2, 2.85), (qx + QW / 2, 2.37), color=MUTED)
tag(ax, qx + QW / 2 + 0.28, 2.61, "是", MUTED)

arrow(ax, (qx + QW, 2.01), (7.15, 2.01), color=PALETTE[1])
tag(ax, 6.95, 2.20, "是", PALETTE[1])

arrow(ax, (qx + QW / 2, 1.65), (qx + QW / 2, 1.07), color=PALETTE[3])
tag(ax, qx + QW / 2 + 0.28, 1.36, "否", PALETTE[3])

note(ax, 5.0, 0.08, "能用 workflow 解决的，不要上 agent", MUTED, fontsize=7.4)

save(fig, "l17-decision-tree")
