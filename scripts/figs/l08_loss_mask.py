"""L8-1：SFT 的 loss mask。

一条对话被拼成一条序列，但只有 assistant 的 token 参与损失。
system 与 user 部分只作为条件，不产生梯度。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 2.8, xmax=10, ymax=4.2)

segs = [
    ("<|system|>", 0, False),
    ("你是助手", 0, False),
    ("<|user|>", 0, False),
    ("北京天气", 0, False),
    ("<|assistant|>", 1, True),
    ("今天晴", 1, True),
    ("26 度", 1, True),
    ("<|end|>", 1, True),
]

x = 0.35
y = 2.55
h = 0.62
for text, mask, _ in segs:
    w = 0.30 + 0.115 * len(text)
    color = PALETTE[1] if mask else PALETTE[4]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=0.16, edgecolor=color, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK)
    if mask:
        ax.text(
            x + w / 2,
            y - 0.22,
            "$\\checkmark$",
            ha="center",
            va="center",
            fontsize=7,
            color=PALETTE[1],
        )
    x += w + 0.08

note(ax, 2.0, 3.55, "条件部分：不计算损失", PALETTE[4], fontsize=7)
note(ax, 7.2, 3.55, "回答部分：计算损失", PALETTE[1], fontsize=7)

ax.plot([0.35, 9.65], [1.95, 1.95], color=MUTED, lw=0.8, ls=":")
note(ax, 5.0, 1.62, "同一条序列，两种角色：loss mask 决定哪些位置贡献梯度", MUTED, fontsize=6.8)
note(
    ax,
    5.0,
    1.15,
    "若整条都算损失，模型会把「用户说了什么」也当成要学的内容",
    PALETTE[1],
    fontsize=6.6,
)
note(
    ax,
    5.0,
    0.70,
    "有效监督比例 = 回答 token 数 / 总 token 数，通常只有 30%–50%",
    MUTED,
    fontsize=6.6,
)

save(fig, "l08-loss-mask")
