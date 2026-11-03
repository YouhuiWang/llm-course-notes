"""L12-4：三种注意力模式的可见范围。

全注意力是 $O(n^2)$；滑窗只保留最近 $w$ 个；稀疏/块状只保留一部分位置。
颜色越深表示越"能看见"。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, save, setup

setup()

n = 24
w = 6

full = np.ones((n, n))
window = np.zeros((n, n))
for i in range(n):
    window[i, max(0, i - w + 1): i + 1] = 1

sparse = np.zeros((n, n))
for i in range(n):
    sparse[i, i] = 1
    for j in range(0, n, 4):
        if j <= i:
            sparse[i, j] = 1
    if i - 1 >= 0:
        sparse[i, i - 1] = 1

panels = [
    (full, "(a) 全注意力：$O(n^2)$", "精确但最贵"),
    (window, f"(b) 滑窗（w={w}）：$O(nw)$", "局部精确，远处丢失"),
    (sparse, "(c) 稀疏/块状：$O(n\\sqrt{n})$", "折中，实现复杂"),
]

fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
for ax, (mat, title, desc) in zip(axes, panels):
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.3, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=8.4)
    ax.text(n / 2, n + 1.2, desc, ha="center", fontsize=6.6, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.text(0.5, -0.05, "纵轴 = 查询位置，横轴 = 被看的位置；深色表示参与注意力",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l12-attention-modes")
