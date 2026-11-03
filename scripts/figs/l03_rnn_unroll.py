"""L3-1：RNN 按时间展开。

同一个循环单元在时间上被复用：每个时刻都吃当前输入 $x_t$ 与上一时刻的
隐状态 $h_{t-1}$，吐出 $h_t$。**权重在所有时刻共享**，这是 RNN 参数少的原因，
也是它无法并行训练的原因。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.4)

W, H = 1.05, 0.62
ys = 2.15
xs = [0.45, 2.55, 4.65, 6.75, 8.55]

for i, x in enumerate(xs):
    if i == 4:
        break
    # 输入
    box(ax, x, 0.55, W, H, f"$x_{i+1}$", PALETTE[2], fontsize=7.4, fill_alpha=0.12)
    # 隐状态
    box(ax, x, ys, W, H, f"$h_{i+1}$", PALETTE[0], fontsize=7.4, fill_alpha=0.14)
    # 输出
    box(ax, x, 3.55, W, H, f"$y_{i+1}$", PALETTE[1], fontsize=7.4, fill_alpha=0.12)

    arrow(ax, (x + W / 2, 1.17), (x + W / 2, ys), color=MUTED)
    arrow(ax, (x + W / 2, ys + H), (x + W / 2, 3.55), color=MUTED)

    # 时间轴递推
    if i < 3:
        arrow(ax, (x + W, ys + H / 2), (xs[i + 1], ys + H / 2), color=PALETTE[0], lw=1.6)

note(ax, 5.0, 4.25, "同一个单元，权重 $W_{xh}, W_{hh}, W_{hy}$ 在所有时刻共享", MUTED, fontsize=7)
note(ax, 5.0, 0.15, "展开后是一条链：$h_t$ 依赖 $h_{t-1}$，无法并行", PALETTE[0], fontsize=6.8)

save(fig, "l03-rnn-unroll")
