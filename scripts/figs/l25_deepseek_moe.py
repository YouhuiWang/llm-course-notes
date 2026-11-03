"""L25-2：DeepSeek 的 MoE 结构。

两类专家：共享专家对所有 token 都激活（承载通用知识），
细粒度专家每个 token 只选少数几个（承载专门知识）。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.2)

box(ax, 0.25, 2.35, 0.9, 0.55, "token", PALETTE[4], fontsize=6.8)
box(ax, 1.55, 2.35, 1.2, 0.55, "Router", PALETTE[0], fontsize=6.8)
arrow(ax, (1.15, 2.62), (1.55, 2.62), color=MUTED)

# 共享专家
box(ax, 3.35, 3.05, 1.5, 0.55, "共享专家\n（始终激活）", PALETTE[2], fontsize=6.0)
arrow(ax, (2.75, 2.75), (3.35, 3.32), color=PALETTE[2], lw=1.2)

# 细粒度专家
for i in range(6):
    x = 3.35 + i * 0.72
    box(ax, x, 1.55, 0.62, 0.55, f"E{i + 1}", PALETTE[1], fontsize=6.0, fill_alpha=0.10)
    if i in (0, 2, 5):
        arrow(ax, (2.75, 2.45), (x + 0.31, 2.10), color=PALETTE[1], lw=0.9)

box(ax, 8.15, 2.35, 1.55, 0.55, "加权求和", PALETTE[3], fontsize=6.6)
arrow(ax, (4.85, 3.32), (8.15, 2.80), color=PALETTE[2], lw=0.9)
# 只连被选中的三个专家（E1/E3/E6），避免箭头穿过其它方块
for i in (0, 2, 5):
    ax.plot([3.35 + i * 0.72 + 0.31, 3.35 + i * 0.72 + 0.31], [2.10, 1.92], color=MUTED, lw=0.6)
arrow(ax, (3.35 + 5 * 0.72 + 0.31, 1.92), (8.15, 2.45), color=PALETTE[1], lw=0.9)

note(ax, 5.0, 1.05, "V3：1 个共享专家 + 256 个细粒度专家，每 token 选 8 个", MUTED, fontsize=6.8)
note(
    ax,
    5.0,
    0.62,
    "共享专家承载通用知识，细粒度专家承载专门知识——减少冗余",
    PALETTE[2],
    fontsize=6.6,
)
note(ax, 5.0, 0.22, "671B 总参数，每 token 只激活 37B（5.5%）", PALETTE[1], fontsize=6.6)

save(fig, "l25-deepseek-moe")
