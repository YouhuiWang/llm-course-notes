"""L23-2：数据污染。

如果测试题在训练集里出现过，分数就不可信。
检测方法：算测试样本与训练语料的最长公共 n-gram 重叠。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

rng = np.random.default_rng(7)
clean = rng.normal(0.12, 0.05, 800)
contaminated = rng.normal(0.72, 0.10, 200)

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.hist(clean, bins=40, color=PALETTE[0], alpha=0.75, label="正常样本（未污染）")
ax.hist(contaminated, bins=40, color=PALETTE[1], alpha=0.75, label="疑似污染样本")

ax.axvline(0.5, color=MUTED, lw=0.9, ls="--")
ax.text(0.52, 60, "判定阈值 0.5", fontsize=6.8, color=MUTED)

ax.set_xlabel("与训练语料的 13-gram 重叠率", fontsize=8)
ax.set_ylabel("样本数", fontsize=8)
ax.set_title("污染检测：双峰分布意味着测试集被泄漏", loc="left", fontsize=9.5)
ax.legend(fontsize=7.2)
clean_axes(ax)

fig.text(0.5, -0.03, "阈值以上的样本要从评测里剔除，或单独报告",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l23-contamination")
