"""L20-1：四种测试时计算方式。

同样的推理预算，可以花在"想得更长""多采几次""搜索分支""反复修改"上。
四条路的收益曲线完全不同。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 2.8, xmax=10, ymax=4.2)

items = [
    ("① 更长 CoT", "一次前向，\n多写中间步骤", "线性增加 token", PALETTE[0]),
    ("② 并行采样", "采 k 次，\n多数投票", "$k\\times$ 推理成本", PALETTE[2]),
    ("③ 搜索", "树状展开，\n按价值剪枝", "指数级候选", PALETTE[3]),
    ("④ 顺序精修", "基于反馈\n逐步改写", "多轮前向", PALETTE[1]),
]

W, H = 2.2, 1.55
for i, (name, how, cost, color) in enumerate(items):
    x = 0.2 + i * 2.42
    ax.add_patch(
        __import__("matplotlib").patches.FancyBboxPatch(
            (x, 1.55),
            W,
            H,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            lw=1.1,
        )
    )
    ax.text(x + W / 2, 2.85, name, ha="center", fontsize=7.6, color=color)
    ax.text(x + W / 2, 2.22, how, ha="center", fontsize=6.2, color=INK, linespacing=1.5)
    ax.text(x + W / 2, 1.72, cost, ha="center", fontsize=5.8, color=MUTED)

note(ax, 5.0, 0.95, "四条路的共同前提：模型本身有能力，只是没「想透」", MUTED, fontsize=6.8)
note(
    ax,
    5.0,
    0.50,
    "若模型根本不会，再多算力也变不出来——这是测试时扩展的硬边界",
    PALETTE[1],
    fontsize=6.6,
)

save(fig, "l20-strategies")
