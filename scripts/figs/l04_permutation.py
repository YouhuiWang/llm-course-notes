"""L4-4：置换等变的直观演示。

"狗咬人" 换成 "人咬狗"，注意力矩阵的数值一个没变，只是行列换了顺序。
模型看到的仍然是同一张表，所以它区分不了这两句话。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from style import INK, MUTED, PALETTE, save, setup

setup()

# 构造一个非对称的注意力权重矩阵（3 个 token）
W = np.array(
    [
        [0.62, 0.31, 0.07],
        [0.18, 0.55, 0.27],
        [0.09, 0.36, 0.55],
    ]
)

# 交换第 1 个与第 3 个 token：狗咬人 → 人咬狗
P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
W_perm = P @ W @ P.T

tokens_a = ["狗", "咬", "人"]
tokens_b = ["人", "咬", "狗"]

cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#cfe0ee", "#5d8fb5", "#1f4e6b"])

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))


def draw(ax, mat, tokens, title):
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.7)
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if v > 0.45 else INK,
            )
    ax.set_xticks(range(3), tokens)
    ax.set_yticks(range(3), tokens)
    ax.set_xlabel("键", fontsize=8, color=MUTED)
    ax.set_ylabel("查询", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


draw(axes[0], W, tokens_a, "(a) 狗 咬 人")
draw(axes[1], W_perm, tokens_b, "(b) 人 咬 狗（交换首尾）")



fig.suptitle("没有位置编码时：注意力对置换等变", fontsize=10.5)
save(fig, "l04-permutation")
