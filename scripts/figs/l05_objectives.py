"""L5-3：四种预训练目标的可见性差异。

CLM 只能看左边；MLM 双向但只还原被遮的词；span corruption 一次遮一段；
prefix-LM 前缀双向、后缀因果。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import box, canvas, note

setup()

fig, ax = canvas(7.0, 3.8, xmax=10.0, ymax=5.4)

tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "slept"]
W, H, STEP = 0.92, 0.48, 1.10
x0 = 0.55

rows = [
    ("CLM\n（GPT）", [0] * 8, 4.55, "每个位置预测下一个词，只能看左边"),
    ("MLM\n（BERT）", [0, 1, 0, 0, 1, 0, 0, 1], 3.25, "随机遮 15%，用左右两边一起还原"),
    ("Span\n（T5）", [0, 1, 1, 1, 0, 1, 1, 0], 1.95, "连续遮一段，解码器逐段还原"),
    ("Prefix\n（GLM）", [0, 0, 0, 1, 1, 1, 1, 1], 0.65, "前 3 个双向，后面因果"),
]

for label, mask, y, desc in rows:
    for i, t in enumerate(tokens):
        masked = mask[i] == 1
        color = PALETTE[1] if masked else PALETTE[0]
        box(ax, x0 + i * STEP, y, W, H,
            "$\\langle M\\rangle$" if masked else t,
            color, fontsize=6.2 if masked else 6.6, fill_alpha=0.16 if masked else 0.08)
    ax.text(0.05, y + H / 2, label, fontsize=6.6, color=INK, va="center", ha="left")
    note(ax, 1.75, y - 0.24, desc, MUTED, fontsize=6.2, ha="left")

note(ax, 5.0, 0.12, "颜色深 = 被遮住的位置；同一种模型换个遮法就是另一种预训练目标", MUTED, fontsize=6.8)

save(fig, "l05-objectives")
