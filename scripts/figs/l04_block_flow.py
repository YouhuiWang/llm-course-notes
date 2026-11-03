"""L4-4：Transformer block 的数据流（Pre-LN 结构）。

两行布局，避免单行过宽导致字太小；残差通路用弧线表示。
"""

from __future__ import annotations

from style import PALETTE, MUTED, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(6.8, 3.0, xmax=10, ymax=4.4)

W, H = 1.42, 0.62
y1, y2 = 3.30, 1.55
xs = [0.20, 1.95, 3.70, 5.45, 7.20]

# ---- 第一行 ----
box(ax, xs[0], y1, W, H, "$x$\n$(n\\times d)$", PALETTE[0])
box(ax, xs[1], y1, W, H, "RMSNorm", PALETTE[1])
box(ax, xs[2], y1, W, H, "Q / K / V\n投影", PALETTE[1])
box(ax, xs[3], y1, W, H, "缩放点积\n注意力", PALETTE[2])
box(ax, xs[4], y1, W, H, "$W_O$\n投影", PALETTE[1])

for i in range(4):
    arrow(ax, (xs[i] + W, y1 + H / 2), (xs[i + 1], y1 + H / 2), color=MUTED)

# ---- 第一行残差 ----
ax_plus1 = 9.0
box(ax, ax_plus1 - 0.22, y1 + H / 2 - 0.22, 0.44, 0.44, "$\\oplus$", PALETTE[0], fontsize=10)
arrow(ax, (xs[4] + W, y1 + H / 2), (ax_plus1 - 0.24, y1 + H / 2), color=MUTED)
arrow(ax, (xs[0] + W / 2, y1), (xs[0] + W / 2, 2.42), color=PALETTE[0], dashed=True, style="-")
arrow(
    ax,
    (xs[0] + W / 2, 2.42),
    (ax_plus1, 2.42),
    color=PALETTE[0],
    dashed=True,
    style="-",
)
arrow(ax, (ax_plus1, 2.42), (ax_plus1, y1 + H / 2 - 0.22), color=PALETTE[0], dashed=True)
note(ax, 4.7, 2.62, "残差通路：恒等映射，梯度直通", PALETTE[0], fontsize=7)

# ---- 第二行 ----
box(ax, xs[0], y2, W, H, "RMSNorm", PALETTE[1])
box(ax, xs[1] + 0.3, y2, W + 0.35, H, "SwiGLU FFN\n$d \\to 2.7d \\to d$", PALETTE[3], fontsize=7.4)
box(ax, xs[3], y2, 1.9, H, "$\\oplus$", PALETTE[0], fontsize=10)
box(ax, xs[3] + 2.3, y2, W, H, "输出\n$(n\\times d)$", PALETTE[0])

# 从第一行的 ⊕ 下到第二行
arrow(ax, (ax_plus1, y1 + H / 2 - 0.24), (ax_plus1, 2.98), color=MUTED, style="-")
arrow(ax, (ax_plus1, 2.98), (ax_plus1, y2 + H + 0.35), color=MUTED, style="-")
arrow(ax, (ax_plus1, y2 + H + 0.35), (xs[0] + W / 2, y2 + H + 0.35), color=MUTED, style="-")
arrow(ax, (xs[0] + W / 2, y2 + H + 0.35), (xs[0] + W / 2, y2 + H), color=MUTED)

arrow(ax, (xs[0] + W, y2 + H / 2), (xs[1] + 0.3, y2 + H / 2), color=MUTED)
arrow(ax, (xs[1] + 0.3 + W + 0.35, y2 + H / 2), (xs[3], y2 + H / 2), color=MUTED)
arrow(ax, (xs[3] + 1.9, y2 + H / 2), (xs[3] + 2.3, y2 + H / 2), color=MUTED)

# 第二行残差：从 ⊕1 的输出绕到 ⊕2（绕开 FFN）
arrow(ax, (xs[0], y2 + H / 2), (0.05, y2 + H / 2), color=PALETTE[0], dashed=True, style="-")
arrow(ax, (0.05, y2 + H / 2), (0.05, 0.95), color=PALETTE[0], dashed=True, style="-")
arrow(ax, (0.05, 0.95), (xs[3] + 0.95, 0.95), color=PALETTE[0], dashed=True, style="-")
arrow(ax, (xs[3] + 0.95, 0.95), (xs[3] + 0.95, y2), color=PALETTE[0], dashed=True)
note(ax, 3.2, 0.72, "第二条残差：绕过 FFN", PALETTE[0], fontsize=7)

note(ax, 5.0, 0.22, "Pre-LN：归一化放在子层入口，残差通路保持干净", MUTED, fontsize=7.6)

save(fig, "l04-block-flow")
