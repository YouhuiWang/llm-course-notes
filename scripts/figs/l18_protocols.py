"""L18-2：MCP 与 A2A 的分工。

MCP 解决"agent 怎么连工具"，A2A 解决"agent 怎么找 agent"。
两者不是替代关系，而是不同层次。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.4)

# 上层：agent 之间
ax.add_patch(Rectangle((0.2, 2.55), 9.6, 1.35, facecolor=PALETTE[2], alpha=0.07, edgecolor="none"))
note(ax, 0.45, 3.68, "A2A 层：agent ↔ agent", PALETTE[2], fontsize=7.2, ha="left")
for i, name in enumerate(["规划 agent", "检索 agent", "编码 agent"]):
    box(ax, 1.2 + i * 2.9, 2.75, 1.85, 0.62, name, PALETTE[2], fontsize=6.6)
    if i < 2:
        arrow(ax, (3.05 + i * 2.9, 3.06), (4.10 + i * 2.9, 3.06), color=PALETTE[2], style="<|-|>")

# 下层：agent 连工具
ax.add_patch(Rectangle((0.2, 0.55), 9.6, 1.55, facecolor=PALETTE[0], alpha=0.06, edgecolor="none"))
note(ax, 0.45, 1.95, "MCP 层：agent ↔ 工具/数据", PALETTE[0], fontsize=7.2, ha="left")
for i, name in enumerate(["文件系统", "数据库", "Web API"]):
    box(ax, 1.2 + i * 2.9, 0.75, 1.85, 0.62, name, PALETTE[0], fontsize=6.6)

arrow(ax, (2.1, 2.75), (2.1, 1.37), color=MUTED, lw=0.8)
note(
    ax,
    5.0,
    0.25,
    "MCP 管「能做什么」，A2A 管「找谁做」——协议层次不同，不能互相替代",
    MUTED,
    fontsize=6.8,
)

save(fig, "l18-protocols")
