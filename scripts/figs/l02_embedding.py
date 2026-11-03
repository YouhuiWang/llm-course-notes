"""L2-4：词向量空间里的语义算术。

把 300 维压到 2 维后，"国王 − 男人 + 女人 ≈ 王后"这类关系
表现为**方向相同、长度相近的平移**。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

pts = {
    "男人": (1.0, 1.2),
    "女人": (1.0, 2.6),
    "国王": (3.2, 1.2),
    "王后": (3.2, 2.6),
    "王子": (4.6, 1.2),
}

fig, ax = plt.subplots(figsize=(6.4, 3.4))

for word, (x, y) in pts.items():
    ax.scatter([x], [y], s=36, color=PALETTE[0], zorder=5)
    ax.annotate(word, xy=(x, y), xytext=(x + 0.06, y + 0.12), fontsize=8, color=INK)

# 平移箭头：男人→国王，女人→王后
for a, b, c in [("男人", "国王", PALETTE[1]), ("女人", "王后", PALETTE[1])]:
    xa, ya = pts[a]
    xb, yb = pts[b]
    ax.annotate(
        "",
        xy=(xb, yb),
        xytext=(xa, ya),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, ls="--"),
    )

# 向量加法示意：国王 - 男人 + 女人
xa, ya = pts["男人"]
xb, yb = pts["国王"]
ax.annotate(
    "",
    xy=(xb - xa + pts["女人"][0], yb - ya + pts["女人"][1]),
    xytext=pts["女人"],
    arrowprops=dict(arrowstyle="-|>", color=PALETTE[2], lw=1.6),
)
ax.annotate(
    "国王 − 男人 + 女人",
    xy=(pts["王后"][0], pts["王后"][1] - 0.1),
    xytext=(1.1, 3.35),
    fontsize=7.4,
    color=PALETTE[2],
    arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=0.8),
)

ax.set_xlim(0.5, 5.4)
ax.set_ylim(0.6, 3.7)
ax.set_xlabel("维度 1", fontsize=7.5)
ax.set_ylabel("维度 2", fontsize=7.5)
ax.set_title("语义关系 = 空间中的固定方向", loc="left", fontsize=9.5)
ax.set_xticks([])
ax.set_yticks([])
clean_axes(ax, grid_axis=None)

fig.text(
    0.5,
    -0.03,
    "「性别」是同一方向，「王室」是另一方向——两个方向叠加就得到王后",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l02-embedding")
