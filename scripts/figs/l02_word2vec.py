"""L2-3：CBOW 与 Skip-gram 的结构差异。

CBOW 用多个上下文词预测中心词；Skip-gram 反过来，用中心词预测周围的词。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.2, xmax=10, ymax=4.8)

# ---------- 左：CBOW ----------
note(ax, 2.4, 4.35, "CBOW：上下文 → 中心词", PALETTE[0], fontsize=8)
ctx = ["我", "在", "看"]
for i, w in enumerate(ctx):
    box(ax, 0.35, 2.75 - i * 0.95, 0.95, 0.6, w, PALETTE[0], fontsize=7.6)
    arrow(ax, (1.30, 3.05 - i * 0.95), (1.85, 2.35), color=MUTED)

box(ax, 1.85, 1.95, 1.0, 0.7, "平均\n投影", PALETTE[2], fontsize=6.8)
arrow(ax, (2.85, 2.30), (3.30, 2.30), color=MUTED)
box(ax, 3.30, 1.95, 0.95, 0.7, "书", PALETTE[1], fontsize=7.6)

# ---------- 右：Skip-gram ----------
note(ax, 7.4, 4.35, "Skip-gram：中心词 → 上下文", PALETTE[1], fontsize=8)
box(ax, 5.75, 1.95, 0.95, 0.7, "书", PALETTE[1], fontsize=7.6)
arrow(ax, (6.70, 2.30), (7.15, 2.30), color=MUTED)
box(ax, 7.15, 1.95, 0.85, 0.7, "投影", PALETTE[2], fontsize=6.8)

for i, w in enumerate(ctx):
    box(ax, 8.55, 2.75 - i * 0.95, 0.95, 0.6, w, PALETTE[0], fontsize=7.6)
    arrow(ax, (8.00, 2.30), (8.55, 3.05 - i * 0.95), color=MUTED)

note(
    ax,
    5.0,
    0.45,
    "CBOW 训练快、对高频词友好；Skip-gram 对低频词更好，是 word2vec 论文的默认选择",
    MUTED,
    fontsize=6.9,
)

save(fig, "l02-word2vec")
