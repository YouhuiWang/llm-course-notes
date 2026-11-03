"""L5-2：四种可见性矩阵。

encoder 全可见；decoder 只能看左边；交叉注意力里目标能看全部源；
prefix-LM 前缀双向、后缀因果。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, save, setup

setup()

n = 6
full = np.ones((n, n))
causal = np.tril(np.ones((n, n)))
prefix = np.tril(np.ones((n, n)))
prefix[:3, :] = 1.0  # 前 3 个位置双向可见

panels = [
    (full, "(a) encoder-only：全可见", "BERT"),
    (causal, "(b) decoder-only：因果", "GPT"),
    (full, "(c) 交叉注意力：目标看全部源", "T5 解码器"),
    (prefix, "(d) prefix-LM：前缀双向", "GLM / UniLM"),
]

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4))
for ax, (mat, title, tag) in zip(axes.ravel(), panels):
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.35, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=6.5)
    ax.set_yticklabels([str(i + 1) for i in range(n)], fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=8)
    ax.text(n - 0.4, -0.9, tag, fontsize=6.8, color=MUTED, ha="right")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("键（被看的位置）", fontsize=6.6, color=MUTED)
    ax.set_ylabel("查询（当前位置）", fontsize=6.6, color=MUTED)

fig.suptitle("四种可见性：颜色越深越能看", fontsize=10)
save(fig, "l05-masks")
