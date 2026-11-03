"""L16-4：RAGAS 的四个指标。

两个看检索（上下文精度/召回），两个看生成（忠实度/答案相关性）。
分开看才能定位问题出在哪一段。
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.4)

items = [
    ("上下文精度", "检索到的片段里\n有多少是相关的", "检索", PALETTE[0], 0.62, 2.55),
    ("上下文召回", "需要的片段\n有多少被检索到", "检索", PALETTE[0], 5.55, 2.55),
    ("忠实度", "答案里的每句话\n是否有上下文支撑", "生成", PALETTE[2], 0.62, 1.15),
    ("答案相关性", "答案是否真的\n回答了问题", "生成", PALETTE[2], 5.55, 1.15),
]

W, H = 3.85, 1.15
for name, body, stage, color, x, y in items:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            W,
            H,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            lw=1.1,
        )
    )
    ax.text(x + 0.2, y + H - 0.28, name, fontsize=8.0, color=color, ha="left")
    ax.text(x + 0.2, y + 0.36, body, fontsize=6.4, color=INK, ha="left", linespacing=1.5)
    ax.text(x + W - 0.2, y + H - 0.28, stage, fontsize=6.2, color=MUTED, ha="right")

note(ax, 5.0, 3.85, "检索指标决定上限，生成指标决定兑现", MUTED, fontsize=7.2)
note(
    ax,
    5.0,
    0.65,
    "排查顺序：先看上下文召回（找不到就一定答错），再看忠实度（找到却编了）",
    PALETTE[1],
    fontsize=6.8,
)

save(fig, "l16-ragas")
