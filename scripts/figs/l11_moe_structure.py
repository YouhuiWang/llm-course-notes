"""L11-1：MoE 的基本结构。

每个 token 先过路由算一组权重，只激活 top-k 个专家，
再把它们的输出按权重加权求和。总参数很多，但每个 token 只算一小部分。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.2, xmax=10, ymax=4.6)

# 输入 token
for i in range(3):
    box(ax, 0.20, 3.05 - i * 0.85, 0.85, 0.55, f"$x_{i + 1}$", PALETTE[4], fontsize=7.2)

# 路由
box(
    ax,
    1.55,
    2.05,
    1.25,
    1.55,
    "Router\n$\\mathrm{softmax}(W_g x)$\n取 top-2",
    PALETTE[0],
    fontsize=6.4,
)
for i in range(3):
    arrow(ax, (1.05, 3.32 - i * 0.85), (1.55, 2.85), color=MUTED, lw=0.8)

# 专家
experts = [("专家 1", 3.35), ("专家 2", 4.55), ("专家 3", 5.75), ("专家 4", 6.95)]
for name, x in experts:
    box(ax, x, 2.35, 1.0, 1.0, name, PALETTE[2], fontsize=6.8, fill_alpha=0.10)
    arrow(ax, (2.80, 2.82), (x, 2.85), color=MUTED, lw=0.7)

# 加权求和
box(ax, 8.20, 2.35, 1.05, 1.0, "$\\sum g_i E_i$\n输出", PALETTE[1], fontsize=6.6)
# 只有被选中的两个专家连到输出，避免箭头穿过其它专家
for _, x in [experts[0], experts[2]]:
    arrow(ax, (x + 1.0, 2.85), (8.20, 2.85), color=PALETTE[2], lw=1.0)
for _, x in [experts[1], experts[3]]:
    ax.plot([x + 0.5, x + 0.5], [2.35, 2.15], color=MUTED, lw=0.8, ls=":")

note(
    ax, 5.0, 1.45, "每个 token 只走 2 个专家：计算量按激活参数算，显存按总参数算", MUTED, fontsize=7
)
note(ax, 5.0, 0.95, "灰色箭头是路由打分；实际只有 top-k 条通路被激活", MUTED, fontsize=6.6)
note(
    ax,
    5.0,
    0.45,
    "难点不在结构，在「负载均衡」——路由很容易只挑同一个专家",
    PALETTE[1],
    fontsize=6.8,
)

save(fig, "l11-moe-structure")
