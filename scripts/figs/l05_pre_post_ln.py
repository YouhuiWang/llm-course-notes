"""L5-1：Pre-LN 与 Post-LN 的数据流对比。

Post-LN 把归一化放在残差相加之后，梯度回传时必须穿过 LN；
Pre-LN 放在子层入口，残差通路是一条干净的直线——这是深层 Transformer 能训起来的关键。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.4, 3.4, xmax=11.4, ymax=5.0)


def draw(x0, title, pre_ln: bool, color):
    W, H = 0.98, 0.5
    xs = [x0, x0 + 1.30, x0 + 2.60, x0 + 3.90]
    y = 3.35

    ax.text(x0 + 2.45, 4.55, title, fontsize=8.4, color=color, ha="center")

    if pre_ln:
        seq = ["$x$", "LN", "MHA", "$+$"]
    else:
        seq = ["$x$", "MHA", "$+$", "LN"]

    for i, name in enumerate(seq):
        box(ax, xs[i], y, W, H, name, color, fontsize=7.6)
        if i:
            arrow(ax, (xs[i - 1] + W, y + H / 2), (xs[i], y + H / 2), color=MUTED)

    # 残差：从 x 直接连到 ⊕
    plus_x = xs[3] + W / 2 if pre_ln else xs[2] + W / 2
    ax.plot([xs[0] + W / 2, xs[0] + W / 2], [y, y - 0.75], color=PALETTE[0], lw=1.3, ls="--")
    ax.plot([xs[0] + W / 2, plus_x], [y - 0.75, y - 0.75], color=PALETTE[0], lw=1.3, ls="--")
    arrow(ax, (plus_x, y - 0.75), (plus_x, y), color=PALETTE[0], dashed=True)

    note(
        ax,
        x0 + 2.45,
        1.95,
        "残差干净" if pre_ln else "残差穿过 LN",
        PALETTE[0] if pre_ln else PALETTE[1],
        fontsize=6.8,
    )


draw(0.30, "Post-LN（原版 Transformer）", False, PALETTE[1])
draw(6.10, "Pre-LN（现代大模型）", True, PALETTE[0])

note(
    ax,
    5.7,
    0.75,
    "Post-LN 在 12 层以上就要靠 warmup 才能训；Pre-LN 可以直接上 100 层",
    MUTED,
    fontsize=7,
)
note(ax, 5.7, 0.30, "代价：Pre-LN 的最终层输出未归一化，推理时通常再补一个 LN", MUTED, fontsize=6.8)

save(fig, "l05-pre-post-ln")
