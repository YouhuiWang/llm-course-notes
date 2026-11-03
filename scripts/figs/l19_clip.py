"""L19-2：CLIP 的对比学习。

一个 batch 里 N 对图文，算 N×N 的相似度矩阵，对角线是正样本，
其余全是负样本——一个 batch 就提供 N² 个训练信号。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup

setup()

sim = np.array(
    [
        [0.92, 0.31, 0.18],
        [0.25, 0.88, 0.30],
        [0.20, 0.28, 0.90],
    ]
)

cmap = LinearSegmentedColormap.from_list("clip", ["#ffffff", "#cfe0ee", "#5d8fb5", "#1f4e6b"])

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.imshow(sim, cmap=cmap, vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            f"{sim[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if sim[i, j] > 0.6 else INK,
        )
        if i == j:
            ax.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=PALETTE[1], lw=2)
            )

ax.set_xticks(range(3), ["文本 1", "文本 2", "文本 3"], fontsize=7.5)
ax.set_yticks(range(3), ["图像 1", "图像 2", "图像 3"], fontsize=7.5)
ax.set_xlabel("文本编码", fontsize=7.5, color=MUTED)
ax.set_ylabel("图像编码", fontsize=7.5, color=MUTED)
ax.set_title("对比学习：对角线是正样本", loc="left", fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

fig.text(
    0.5,
    -0.06,
    "每行 softmax 后，正样本概率越高越好；其余 N−1 个都是负样本",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l19-clip")
