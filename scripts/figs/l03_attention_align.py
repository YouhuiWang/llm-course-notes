"""L3-4：seq2seq 里的注意力对齐。

解码"猫"时主要看 "cat"，解码"垫子"时同时看 "the" 和 "mat"——
注意力矩阵把这种对应关系直接画了出来。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from style import INK, MUTED, save, setup

setup()

src = ["The", "cat", "sat", "on", "the", "mat"]
tgt = ["猫", "坐", "在", "垫子", "上"]

raw = np.array(
    [
        [0.10, 0.78, 0.04, 0.02, 0.03, 0.03],  # 猫
        [0.02, 0.10, 0.80, 0.05, 0.01, 0.02],  # 坐
        [0.03, 0.02, 0.12, 0.75, 0.05, 0.03],  # 在
        [0.05, 0.02, 0.02, 0.06, 0.42, 0.43],  # 垫子
        [0.02, 0.01, 0.06, 0.38, 0.08, 0.45],  # 上
    ]
)
w = raw / raw.sum(axis=1, keepdims=True)

cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#cfe0ee", "#5d8fb5", "#1f4e6b"])

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.imshow(w, cmap=cmap, vmin=0, vmax=0.8)

for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        v = w[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=7.2,
            color="white" if v > 0.5 else INK,
        )

ax.set_xticks(range(len(src)), src, fontsize=8)
ax.set_yticks(range(len(tgt)), tgt, fontsize=8)
ax.set_xlabel("源语言（英文，编码器输出）", fontsize=8, color=MUTED)
ax.set_ylabel("目标语言（中文，解码中）", fontsize=8, color=MUTED)
ax.set_title("解码每一步看哪里：注意力权重矩阵", loc="left", fontsize=9.5)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

fig.text(
    0.5,
    -0.02,
    "每行和为 1：解码一个词时，把注意力按权重分配给所有源词",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l03-attention-align")
