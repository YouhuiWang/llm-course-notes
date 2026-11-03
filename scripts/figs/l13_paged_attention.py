"""L13-2：连续分配 vs 分页。

连续分配要为每条请求预留最大长度的空间，长度不一就产生大量碎片；
分页把 KV cache 切成固定大小的块，按需分配，碎片率从 60%–80% 降到 4% 以下。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.2)

# (a) 连续分配
note(ax, 2.4, 3.75, "(a) 连续分配：按最大长度预留", PALETTE[1], fontsize=7.6)
reqs = [("请求 A", 1.2, PALETTE[0]), ("请求 B", 2.6, PALETTE[2]), ("请求 C", 1.8, PALETTE[3])]
y = 3.15
for name, used, color in reqs:
    ax.add_patch(Rectangle((0.4, y), 4.2, 0.42, facecolor="white", edgecolor=MUTED, lw=0.8))
    ax.add_patch(
        Rectangle((0.4, y), used, 0.42, facecolor=color, alpha=0.35, edgecolor=color, lw=0.8)
    )
    ax.text(0.5, y + 0.21, name, fontsize=6.6, va="center", color=INK)
    ax.text(0.4 + used + 0.1, y + 0.21, "预留但空闲", fontsize=6, va="center", color=MUTED)
    y -= 0.62
note(ax, 2.4, 1.05, "碎片率通常 60%–80%", PALETTE[1], fontsize=6.8)

# (b) 分页
note(ax, 7.4, 3.75, "(b) 分页：固定块按需分配", PALETTE[0], fontsize=7.6)
for row in range(3):
    y = 3.15 - row * 0.62
    for col in range(5):
        used = (row == 0 and col < 2) or (row == 1 and col < 3) or (row == 2 and col < 1)
        color = PALETTE[0] if used else "white"
        ax.add_patch(
            Rectangle(
                (5.4 + col * 0.86, y),
                0.8,
                0.42,
                facecolor=color,
                alpha=0.35 if used else 1,
                edgecolor=MUTED,
                lw=0.8,
            )
        )
note(ax, 7.4, 1.05, "按块分配，碎片 < 4%", PALETTE[0], fontsize=6.8)

note(
    ax,
    5.0,
    0.45,
    "同一块物理显存可以被不同请求的不同位置复用——这就是 PagedAttention 的核心",
    MUTED,
    fontsize=6.8,
)

save(fig, "l13-paged-attention")
