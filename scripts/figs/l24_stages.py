"""L24-3：四个阶段与三道闸门。

MVP 只验证"能不能做"，内测验证"好不好用"，灰度验证"稳不稳"，最后才全量。
每道闸门都有明确的通过标准，不达标就退回上一阶段。
"""

from __future__ import annotations

from matplotlib.patches import Polygon

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.2, 2.4, xmax=10, ymax=3.4)

stages = [
    ("MVP", "能不能做", PALETTE[0]),
    ("内测", "好不好用", PALETTE[2]),
    ("灰度", "稳不稳", PALETTE[3]),
    ("全量", "扛不扛得住", PALETTE[1]),
]
gates = ["效果达标", "满意度 > 70%", "错误率 < 1%"]

W = 1.95
for i, (name, body, color) in enumerate(stages):
    x = 0.2 + i * 2.42
    ax.add_patch(
        Polygon(
            [[x, 1.35], [x + W, 1.35], [x + W, 2.25], [x, 2.25]],
            closed=True,
            facecolor=color,
            alpha=0.10,
            edgecolor=color,
            lw=1.0,
        )
    )
    ax.text(x + W / 2, 1.95, name, ha="center", fontsize=7.8, color=color)
    ax.text(x + W / 2, 1.58, body, ha="center", fontsize=6.4, color=INK)

for i, g in enumerate(gates):
    x = 0.2 + (i + 1) * 2.42
    ax.plot([x, x], [1.15, 2.45], color=MUTED, lw=0.8, ls=":")
    ax.text(x, 2.55, g, ha="center", fontsize=5.8, color=MUTED)

note(ax, 5.0, 0.70, "每道闸门都是硬性的：不达标就退回，不带病上线", MUTED, fontsize=6.8)
note(ax, 5.0, 0.35, "灰度阶段要能一键回滚——这是唯一真正的保险", PALETTE[1], fontsize=6.6)

save(fig, "l24-stages")
