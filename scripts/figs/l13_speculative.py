"""L13-4：投机解码。

小模型一次草拟 γ 个 token，大模型一次前向就能验证全部；
接受最长的正确前缀，再从这里继续。数学上与原模型分布完全一致（无损）。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 2.8, xmax=10, ymax=4.0)

box(ax, 0.25, 2.25, 1.5, 0.7, "输入 $x$", PALETTE[4], fontsize=7.2)

# 草稿模型
box(ax, 2.05, 2.25, 1.9, 0.7, "草稿模型（小）\n串行 $\\gamma$ 步", PALETTE[0], fontsize=6.6)
arrow(ax, (1.75, 2.60), (2.05, 2.60), color=MUTED)

# 候选
for i in range(4):
    box(
        ax,
        4.25 + i * 0.62,
        2.35,
        0.52,
        0.5,
        f"$\\hat y_{i + 1}$",
        PALETTE[2],
        fontsize=6.4,
        fill_alpha=0.12,
    )

# 目标模型验证
box(ax, 6.95, 2.25, 1.85, 0.7, "目标模型（大）\n一次前向验证", PALETTE[1], fontsize=6.6)
arrow(ax, (6.85, 2.60), (6.95, 2.60), color=MUTED)

# 接受/拒绝
box(ax, 9.0, 2.25, 0.85, 0.7, "接受\n前 3 个", PALETTE[3], fontsize=6.2)

note(ax, 5.55, 1.55, "4 个候选里接受前 3 个：一次大模型前向产出 3 个 token", MUTED, fontsize=6.8)
note(
    ax,
    5.0,
    1.05,
    "被拒的那个位置按「残差分布」重新采样，保证输出分布不变",
    PALETTE[3],
    fontsize=6.6,
)
note(ax, 5.0, 0.55, "加速比 ≈ 接受长度 / (1 + γ × 草稿代价比)", MUTED, fontsize=6.6)
note(
    ax,
    5.0,
    0.18,
    "草稿模型越准、越大，接受率越高，但草稿本身也越慢——存在最优 γ",
    PALETTE[0],
    fontsize=6.4,
)

save(fig, "l13-speculative")
