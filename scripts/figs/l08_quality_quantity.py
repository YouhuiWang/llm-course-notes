"""L8-2：数据质量与数量的交互。

同样的标注预算，精选数据在 1000 条附近就接近饱和；
随机抓取的数据要到几万条才勉强追平，而且上限更低。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

n = np.logspace(2, 5.7, 200)

curated = 0.82 * (1 - np.exp(-n / 900)) + 0.12
random_ = 0.62 * (1 - np.exp(-n / 12000)) + 0.10

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(n, curated, color=PALETTE[0], lw=2, label="精选数据（人工校验）")
ax.plot(n, random_, color=PALETTE[1], lw=2, ls="--", label="随机抓取（未过滤）")

ax.axhline(0.9, color=MUTED, lw=0.7, ls=":")
ax.annotate("精选 1000 条 ≈ 随机 3 万条",
            xy=(1000, 0.82 * (1 - np.exp(-1000 / 900)) + 0.12),
            xytext=(2600, 0.45),
            fontsize=7,
            color=PALETTE[0],
            arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("训练样本数", fontsize=8)
ax.set_ylabel("人工评测胜率", fontsize=8)
ax.set_title("质量与数量不是线性可换的", loc="left", fontsize=9.5)
ax.set_ylim(0.05, 1.0)
ax.legend(loc="lower right", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "数据清洗的边际收益，远高于多标几千条脏数据",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l08-quality-quantity")
