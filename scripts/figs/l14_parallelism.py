"""L14-1：四种并行。

DP 切数据、TP 切层内矩阵、PP 切层、EP 切专家。
切得越细，通信越频繁——这是选并行策略的核心权衡。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.2, xmax=10, ymax=4.6)

panels = [
    ("数据并行 DP", "每卡一份完整模型\n切分 batch", "梯度 all-reduce", PALETTE[0]),
    ("张量并行 TP", "把矩阵按列/行切开\n层内切分", "每层 all-reduce", PALETTE[2]),
    ("流水并行 PP", "按层切成若干段\n流水执行", "段间点对点", PALETTE[3]),
    ("专家并行 EP", "专家分散到不同卡\ntoken 路由", "all-to-all", PALETTE[1]),
]

W, H = 2.15, 1.45
for i, (name, what, comm, color) in enumerate(panels):
    x = 0.25 + (i % 2) * 4.95
    y = 2.55 - (i // 2) * 1.65
    box(ax, x, y, W, H, "", color, fill_alpha=0.10)
    ax.text(x + W / 2, y + H - 0.28, name, ha="center", fontsize=7.8, color=color)
    ax.text(x + W / 2, y + 0.62, what, ha="center", fontsize=6.2, color=INK, linespacing=1.5)
    ax.text(x + W / 2, y + 0.18, comm, ha="center", fontsize=5.8, color=MUTED)

note(ax, 5.0, 0.35, "切得越细，通信越频繁：TP 每层通信，PP 每段通信，DP 每步通信",
     MUTED, fontsize=6.8)

save(fig, "l14-parallelism")
