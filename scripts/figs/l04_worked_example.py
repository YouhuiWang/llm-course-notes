"""L4-1：三词句子的注意力手算（d_k = 2）。

用最小可算的例子走完"打分 → 归一化 → 加权求和"三步，
每一步都标出具体数字，配合正文的表格逐项对照。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from style import INK, MUTED, PALETTE, save, setup

setup()

tokens = ["我", "爱", "猫"]
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = Q.copy()
V = Q.copy()

scores = Q @ K.T / np.sqrt(2)
Z = np.exp(scores - scores.max(axis=1, keepdims=True))
W = Z / Z.sum(axis=1, keepdims=True)
out = W @ V

cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#cfe0ee", "#5d8fb5", "#1f4e6b"])


def annotate(ax, mat, fmt="{:.2f}", fs=9):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > mat.max() * 0.62 else INK
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color=color)


fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.75]})

# ① 打分
ax = axes[0]
ax.imshow(scores, cmap=cmap, vmin=0, vmax=scores.max())
annotate(ax, scores)
ax.set_xticks(range(3), tokens)
ax.set_yticks(range(3), tokens)
ax.set_xlabel("键（被看的词）", fontsize=8, color=MUTED)
ax.set_ylabel("查询（当前的词）", fontsize=8, color=MUTED)
ax.set_title("① 打分  $q\\cdot k/\\sqrt{d_k}$", loc="left", fontsize=9)
ax.tick_params(length=0)

# ② 归一化
ax = axes[1]
ax.imshow(W, cmap=cmap, vmin=0, vmax=W.max())
annotate(ax, W)
ax.set_xticks(range(3), tokens)
ax.set_yticks(range(3), tokens)
ax.set_xlabel("权重之和 = 1", fontsize=8, color=MUTED)
ax.set_title("② softmax 归一化", loc="left", fontsize=9)
ax.tick_params(length=0)

# ③ 加权求和
ax = axes[2]
ax.imshow(out, cmap=cmap, vmin=0, vmax=out.max())
annotate(ax, out)
ax.set_xticks(range(2), ["维度 1", "维度 2"], fontsize=7.5)
ax.set_yticks(range(3), tokens)
ax.set_title("③ 加权求和 $W V$", loc="left", fontsize=9)
ax.tick_params(length=0)

for ax in axes:
    for s in ax.spines.values():
        s.set_visible(False)


fig.suptitle("三词句子的注意力：从相似度到输出（$d_k=2$）", fontsize=10.5)
save(fig, "l04-worked-example")
