"""L4-4：注意力矩阵长什么样（示意图）。

用一个人工构造的句子演示三类常见模式：
  (a) 因果掩码：只能看左边
  (b) 位置头：关注前一个 token
  (c) 内容头：代词回指先行词
  (d) 注意力汇（attention sink）：大量注意力堆在第一个 token

注意：这是按模式手工构造的示意图，不是真实模型的可视化。
真实模型的可视化见 Lab1 的 `--dump-attention` 输出。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from style import INK, MUTED, save, setup

setup()

tokens = ["小", "猫", "追", "小", "狗", "，", "因为", "它", "很", "调皮", "。"]
n = len(tokens)
causal = np.tril(np.ones((n, n), dtype=bool))

cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#cfe0ee", "#5d8fb5", "#1f4e6b"])


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.where(causal, mat, np.nan)
    row = np.nansum(mat, axis=1, keepdims=True)
    return mat / np.where(row == 0, 1, row)


# (a) 均匀因果：早期层的典型形态
a = np.ones((n, n))

# (b) 位置头：强烈关注前一个 token
b = np.zeros((n, n))
for i in range(n):
    b[i, i - 1] = 3.0 if i > 0 else 0.0
    b[i, i] = 0.4

# (c) 内容头：代词"它"(索引 7) 回指"小猫"(索引 0–1)
c = np.full((n, n), 0.15)
c[7, 0] = c[7, 1] = 4.0
c[7, 8] = 0.6  # 也看"很"

# (d) 注意力汇：首 token 吸收大量注意力
d = np.full((n, n), 0.1)
d[:, 0] = 5.0

panels = [
    (a, "(a) 均匀因果注意力"),
    (b, "(b) 位置头：看前一个 token"),
    (c, "(c) 内容头：代词回指"),
    (d, "(d) 注意力汇：首 token 吸收"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
im = None
for ax, (mat, title) in zip(axes.ravel(), panels):
    m = normalize(mat)
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=np.nanmax(m))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(title, loc="left", fontsize=9)
    ax.tick_params(length=0)
    # 被 mask 的区域画灰
    ax.imshow(np.where(causal, np.nan, 1), cmap="Greys", vmin=0, vmax=2, alpha=0.25)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("键（被关注的 token）", fontsize=7, color=MUTED)
    ax.set_ylabel("查询（当前 token）", fontsize=7, color=MUTED)

cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("注意力权重", fontsize=8, color=INK)
cbar.ax.tick_params(labelsize=7)

fig.suptitle("注意力矩阵的四种典型形态（示意图）", fontsize=11)
save(fig, "l04-attention-patterns")
