"""L17-3：MCP 的 Host–Client–Server 结构。

一个 Host 可以连接多个 Server，每个连接由一个 Client 维护。
"""

from __future__ import annotations

from style import PALETTE, MUTED, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(6.8, 2.7, xmax=10, ymax=4.2)

box(ax, 0.25, 1.45, 2.0, 1.3, "Host\nIDE / 聊天应用\n（决定何时调用）", PALETTE[0], fontsize=7.4)

box(ax, 2.95, 2.55, 1.55, 0.62, "Client A", PALETTE[1], fontsize=7.6)
box(ax, 2.95, 1.05, 1.55, 0.62, "Client B", PALETTE[1], fontsize=7.6)

box(
    ax,
    5.35,
    2.45,
    4.2,
    0.82,
    "Server：文件系统\ntools · resources · prompts",
    PALETTE[2],
    fontsize=7.2,
)
box(
    ax,
    5.35,
    0.95,
    4.2,
    0.82,
    "Server：数据库\ntools · resources · prompts",
    PALETTE[2],
    fontsize=7.2,
)

arrow(ax, (2.25, 2.35), (2.95, 2.86), color=MUTED, style="<|-|>")
arrow(ax, (2.25, 1.85), (2.95, 1.36), color=MUTED, style="<|-|>")
arrow(ax, (4.5, 2.86), (5.35, 2.86), color=MUTED, style="<|-|>")
arrow(ax, (4.5, 1.36), (5.35, 1.36), color=MUTED, style="<|-|>")

note(ax, 1.25, 1.02, "stdio（本地）\n或 HTTP+SSE（远程）", MUTED, fontsize=6.8)
note(
    ax, 5.0, 0.35, "价值在生态：工具方写一次 server，所有支持 MCP 的宿主都能用", MUTED, fontsize=7.2
)

save(fig, "l17-mcp")
