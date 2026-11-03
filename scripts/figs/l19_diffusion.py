"""L19-4：扩散模型的两个过程。

前向：逐步加高斯噪声，直到数据变成纯噪声（无需学习）。
反向：学习一个网络，从纯噪声逐步去噪，恢复出图像。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 2.7, xmax=10, ymax=4.0)

n = 5
for i in range(n):
    alpha = 1 - i / (n - 1) * 0.85
    x = 0.35 + i * 1.05
    ax.add_patch(
        Rectangle(
            (x, 2.35), 0.85, 0.85, facecolor=PALETTE[0], alpha=0.10 + 0.12 * (1 - alpha),
            edgecolor=MUTED, lw=0.7,
        )
    )
    ax.text(x + 0.42, 2.10, f"$x_{{{i * 250}}}$" if i < n - 1 else "$x_T$",
            fontsize=6.4, ha="center", color=MUTED)
    if i < n - 1:
        arrow(ax, (x + 0.85, 2.78), (x + 1.05, 2.78), color=PALETTE[1], lw=1.0)

note(ax, 5.0, 3.55, "前向：加噪（固定，无需学习）", PALETTE[1], fontsize=7)

for i in range(n - 1, 0, -1):
    x = 0.35 + i * 1.05
    ax.add_patch(
        Rectangle(
            (x, 1.05), 0.85, 0.6, facecolor=PALETTE[2], alpha=0.12, edgecolor=MUTED, lw=0.6,
        )
    )
    arrow(ax, (x, 1.65), (x - 0.20, 1.65), color=PALETTE[2], lw=1.0)

note(ax, 5.0, 1.92, "反向：学习去噪（$\\epsilon_\\theta$ 预测噪声）", PALETTE[2], fontsize=7)
note(ax, 5.0, 0.62, "训练目标：让网络预测「这一步加了什么噪声」，均方误差即可", MUTED, fontsize=6.8)
note(ax, 5.0, 0.28, "生成时从纯噪声出发，迭代 $T$ 次得到图像；步数越多越慢", PALETTE[2], fontsize=6.6)

save(fig, "l19-diffusion")
