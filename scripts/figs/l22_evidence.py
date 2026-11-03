"""L22-4：证据分级。

同一个结论，证据强度差别很大。看前沿进展时，先问"证据到哪一级了"。
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.4)

levels = [
    ("L4 生产验证", "在真实业务里跑了半年以上", PALETTE[2]),
    ("L3 独立复现", "至少两个团队独立复现", PALETTE[0]),
    ("L2 基准领先", "在公开 benchmark 上领先", PALETTE[3]),
    ("L1 单篇论文", "作者自己的实验", PALETTE[1]),
]

W, H = 8.4, 0.62
for i, (name, desc, color) in enumerate(levels):
    y = 3.45 - i * 0.78
    ax.add_patch(
        FancyBboxPatch(
            (0.75, y),
            W,
            H,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            lw=1.0,
        )
    )
    ax.text(0.95, y + H / 2, name, fontsize=7.4, color=color, va="center")
    ax.text(2.55, y + H / 2, desc, fontsize=6.4, color=INK, va="center")

note(ax, 5.0, 0.42, "越往上越可信，但数量越少；大多数「突破」停在 L1–L2", MUTED, fontsize=6.8)
note(ax, 5.0, 0.12, "看前沿时先问：这条结论到哪一级了？", PALETTE[1], fontsize=6.6)

save(fig, "l22-evidence")
