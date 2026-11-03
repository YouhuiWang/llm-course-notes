"""L17-2：Manager–Worker 层级规划。

Manager 负责分解与验收，Worker 负责执行；箭头区分"派活"与"汇报"。
"""

from __future__ import annotations

from style import PALETTE, MUTED, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(6.8, 2.9, xmax=10, ymax=4.6)

box(ax, 3.55, 3.35, 2.9, 0.75, "Manager\n拆解目标 · 分配 · 验收", PALETTE[0], fontsize=7.8)

workers = [
    (0.45, "Worker 1\n调研"),
    (2.85, "Worker 2\n实现"),
    (5.25, "Worker 3\n测试"),
    (7.65, "Worker 4\n复核"),
]
# 总线：Manager 与所有 Worker 的公共连接
arrow(ax, (5.0, 3.35), (5.0, 3.02), color=MUTED, style="-")
arrow(ax, (1.40, 3.02), (8.60, 3.02), color=MUTED, style="-")

for x, label in workers:
    box(ax, x, 1.35, 1.9, 0.75, label, PALETTE[2], fontsize=7.6)
    cx = x + 0.95
    arrow(ax, (cx - 0.22, 3.02), (cx - 0.22, 2.10), color=PALETTE[0])
    arrow(ax, (cx + 0.22, 2.10), (cx + 0.22, 3.02), color=PALETTE[2])

note(ax, 0.95, 2.62, "派活", PALETTE[0], fontsize=7)
note(ax, 1.75, 2.62, "汇报", PALETTE[2], fontsize=7)
note(ax, 5.0, 0.55, "分解与执行分离：计划可审查、子任务可并行、失败可单点重试", MUTED, fontsize=7.4)

save(fig, "l17-manager-worker")
