"""L3-3：LSTM 单元的三个门。

遗忘门决定丢多少旧记忆，输入门决定写多少新信息，输出门决定吐多少给下一层。
细胞状态 $c_t$ 沿顶部直通，是梯度能长距离传播的关键。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.4, xmax=10, ymax=5.2)

Y = 4.55  # 细胞状态主线

# 主线
ax.plot([0.35, 8.55], [Y, Y], color=PALETTE[1], lw=2, zorder=1)
ax.text(0.28, Y, "$c_{t-1}$", fontsize=7.6, ha="right", va="center", color=PALETTE[1])
ax.text(8.62, Y, "$c_t$", fontsize=7.6, ha="left", va="center", color=PALETTE[1])

# 运算节点
box(ax, 1.00, Y - 0.21, 0.42, 0.42, "$\\times$", PALETTE[1], fontsize=9)
box(ax, 2.85, Y - 0.21, 0.42, 0.42, "$+$", PALETTE[1], fontsize=9)
box(ax, 4.75, Y - 0.25, 0.95, 0.5, "$\\tanh$", PALETTE[1], fontsize=7.4)
box(ax, 6.55, Y - 0.21, 0.42, 0.42, "$\\times$", PALETTE[1], fontsize=9)

# 门
box(ax, 0.35, 3.05, 1.35, 0.55, "遗忘门 $f_t$", PALETTE[0], fontsize=7)
box(ax, 2.05, 3.05, 1.35, 0.55, "输入门 $i_t$", PALETTE[2], fontsize=7)
box(ax, 3.85, 3.05, 1.35, 0.55, "候选 $g_t$", PALETTE[2], fontsize=7)
box(ax, 5.95, 3.05, 1.35, 0.55, "输出门 $o_t$", PALETTE[3], fontsize=7)

# 门 → 主线
arrow(ax, (1.02, 3.60), (1.20, Y - 0.22), color=PALETTE[0])
arrow(ax, (2.72, 3.60), (2.90, Y - 0.22), color=PALETTE[2])
arrow(ax, (4.52, 3.60), (3.75, Y - 0.22), color=PALETTE[2])
arrow(ax, (6.62, 3.60), (6.75, Y - 0.22), color=PALETTE[3])

# 输出
arrow(ax, (6.97, Y), (7.85, Y), color=PALETTE[3], lw=1.4)
box(ax, 7.85, Y - 0.27, 0.75, 0.54, "$h_t$", PALETTE[3], fontsize=7.6)

# 输入 [h_{t-1}, x_t]
box(ax, 0.30, 1.55, 0.95, 0.48, "$h_{t-1}$", PALETTE[4], fontsize=7)
box(ax, 0.30, 0.85, 0.95, 0.48, "$x_t$", PALETTE[4], fontsize=7)
ax.plot([1.25, 1.45], [1.79, 1.79], color=MUTED, lw=1)
ax.plot([1.25, 1.45], [1.09, 1.09], color=MUTED, lw=1)
ax.plot([1.45, 1.45], [1.09, 1.79], color=MUTED, lw=1)

# 共享输入总线：一条水平线，四个门各自向上取
BUS = 1.35
ax.plot([0.95, 6.62], [BUS, BUS], color=MUTED, lw=1)
arrow(ax, (1.45, 1.44), (1.45, BUS), color=MUTED, lw=1, style="-")
for gx in (1.02, 2.72, 4.52, 6.62):
    arrow(ax, (gx, BUS), (gx, 3.05), color=MUTED, lw=0.9)

note(ax, 4.10, 0.55, "四个门都读同一份输入 $[h_{t-1};\\,x_t]$", MUTED, fontsize=6.8)
note(
    ax,
    4.10,
    0.20,
    "细胞状态只做加法与逐元素乘法，梯度不必穿过 $\\sigma$——这是 LSTM 能记长的原因",
    PALETTE[1],
    fontsize=6.6,
)

save(fig, "l03-lstm")
