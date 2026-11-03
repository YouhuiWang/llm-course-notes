"""L11-4：专家并行。

专家分散在不同 GPU 上，token 要先 all-to-all 发到对应设备，算完再发回来。
通信量正比于"每个 token 要访问几个专家"。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 2.9, xmax=10, ymax=4.2)

gpus = [
    ("GPU 0\nE1 E2", 0.35),
    ("GPU 1\nE3 E4", 2.75),
    ("GPU 2\nE5 E6", 5.15),
    ("GPU 3\nE7 E8", 7.55),
]
for name, x in gpus:
    box(ax, x, 1.55, 1.95, 0.95, name, PALETTE[0], fontsize=7.2, fill_alpha=0.10)

# 上方：all-to-all
ax.annotate(
    "",
    xy=(9.5, 3.35),
    xytext=(0.35, 3.35),
    arrowprops=dict(arrowstyle="-|>", color=PALETTE[1], lw=1.4),
)
ax.annotate(
    "",
    xy=(0.35, 2.95),
    xytext=(9.5, 2.95),
    arrowprops=dict(arrowstyle="-|>", color=PALETTE[2], lw=1.4),
)
note(ax, 4.9, 3.55, "all-to-all：把 token 送到它选中的专家所在设备", PALETTE[1], fontsize=6.8)
note(ax, 4.9, 2.75, "算完再送回来，做加权求和", PALETTE[2], fontsize=6.8)

note(ax, 5.0, 1.15, "通信量 ∝ token 数 × top-k × 隐藏维度", MUTED, fontsize=7)
note(
    ax,
    5.0,
    0.70,
    "top-2 意味着每个 token 要跨设备跑两趟——这是 MoE 训练慢的主要原因",
    MUTED,
    fontsize=6.6,
)
note(
    ax,
    5.0,
    0.25,
    "对策：把 top-k 选中的专家尽量放在同一台机器上（专家亲和调度）",
    PALETTE[0],
    fontsize=6.6,
)

save(fig, "l11-expert-parallel")
