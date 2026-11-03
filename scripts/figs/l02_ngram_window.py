"""L2-1：n-gram 的上下文窗口。

n 越大，预测当前词时能看到的历史越长；但可统计的组合数也随之指数增长。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, save, setup
from diagram import box, canvas, note

setup()

tokens = ["我", "昨天", "在", "图书馆", "看", "了", "书"]
fig, ax = canvas(6.8, 3.2, xmax=10, ymax=5.0)

W, H, STEP = 1.16, 0.6, 1.30
x0 = 0.55
Y = 3.75

for i, t in enumerate(tokens):
    box(ax, x0 + i * STEP, Y, W, H, t, PALETTE[0], fontsize=7.6, fill_alpha=0.12)

# 每行的窗口：只画"括号"，不重复画方块
rows = [
    ("$n=1$", [6], 3.05, "只看当前词"),
    ("$n=2$", [5, 6], 2.35, "看前 1 个词"),
    ("$n=3$", [4, 5, 6], 1.65, "看前 2 个词"),
]

for label, idxs, y, desc in rows:
    xs = x0 + idxs[0] * STEP
    xe = x0 + idxs[-1] * STEP + W
    ax.plot([xs, xe], [y, y], color=PALETTE[1], lw=1.8, solid_capstyle="butt")
    for x in (xs, xe):
        ax.plot([x, x], [y, y + 0.16], color=PALETTE[1], lw=1.8)
    ax.text(0.02, y + 0.08, label, fontsize=7.6, color=INK, ha="left", va="center")
    ax.text(xe + 0.2, y + 0.08, desc, fontsize=6.8, color=MUTED, va="center")

ax.text(x0 + 6 * STEP + W / 2, Y + H + 0.28, "要预测的词", fontsize=6.8,
        color=PALETTE[1], ha="center")

note(ax, 5.0, 0.55,
     "n 每加 1，可能的组合数乘上词表大小：|V|=5 万时，trigram 就有 1.25×10¹⁴ 种",
     MUTED, fontsize=6.9)

save(fig, "l02-ngram-window")
