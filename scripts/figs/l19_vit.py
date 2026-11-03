"""L19-1：ViT 的 patch 化。

把图像切成固定大小的 patch，每个 patch 拉平后过线性层，得到一串 token，
再和文本 token 一样送进 Transformer。图像与语言的架构就此统一。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 2.9, xmax=10, ymax=4.2)

# 图像 4×4 patch
n = 4
cell = 0.42
x0, y0 = 0.35, 1.55
for i in range(n):
    for j in range(n):
        shade = 0.10 + 0.055 * ((i * n + j) % 7)
        ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell), cell - 0.03, cell - 0.03,
                               facecolor=PALETTE[0], alpha=shade, edgecolor=MUTED, lw=0.5))
ax.text(x0 + n * cell / 2, y0 - 0.32, "图像 $224\\times224$", fontsize=6.8, ha="center", color=MUTED)

arrow(ax, (2.30, 2.35), (2.95, 2.35), color=MUTED)
ax.text(2.62, 2.55, "切 patch\n$16\\times16$", fontsize=6.0, ha="center", color=MUTED)

# 线性投影
box(ax, 3.05, 2.00, 1.25, 0.72, "线性投影\n$16\\cdot16\\cdot3\\to d$", PALETTE[1], fontsize=5.8)

arrow(ax, (4.30, 2.36), (4.95, 2.36), color=MUTED)

# token 序列
for i in range(6):
    box(ax, 5.05 + i * 0.68, 2.08, 0.55, 0.55, f"$p_{i + 1}$", PALETTE[2],
        fontsize=6.0, fill_alpha=0.12)
ax.text(6.10, 1.80, "$\\cdots$ 共 196 个 token", fontsize=6.6, color=MUTED)

arrow(ax, (9.15, 2.36), (9.55, 2.36), color=MUTED)
box(ax, 9.05, 1.55, 0.85, 0.6, "Transformer", PALETTE[3], fontsize=5.6)

note(ax, 5.0, 0.95, "图像不再是「特殊输入」，而是另一种 token 序列", MUTED, fontsize=6.8)
note(ax, 5.0, 0.50, "代价：$224/16=14$，$14^2=196$ 个 token，比一句话还长", PALETTE[1], fontsize=6.6)

save(fig, "l19-vit")
