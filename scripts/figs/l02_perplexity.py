"""L2-2：困惑度随 n 的变化——边际收益递减，然后过拟合。

训练集上 n 越大越好；测试集上 n 到 3–4 就到顶，再大反而变差。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

n = np.arange(1, 8)
train = 180 * np.exp(-0.75 * (n - 1)) + 8
test = 180 * np.exp(-0.85 * (n - 1)) + 12 + 2.2 * np.clip(n - 4, 0, None) ** 1.6

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(n, train, "o-", color=PALETTE[0], lw=1.8, ms=4, label="训练集困惑度")
ax.plot(n, test, "s-", color=PALETTE[1], lw=1.8, ms=4, label="测试集困惑度")

best = int(n[np.argmin(test)])
ax.axvline(best, color=MUTED, lw=0.8, ls=":")
ax.annotate(
    f"测试集最优 n={best}",
    xy=(best, test[best - 1]),
    xytext=(best + 0.5, test[best - 1] + 28),
    fontsize=7.4,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

ax.set_xlabel("n（上下文长度）", fontsize=8)
ax.set_ylabel("困惑度（越低越好）", fontsize=8)
ax.set_title("n 越大越好？只在训练集上成立", loc="left", fontsize=9.5)
ax.set_xticks(n)
ax.legend(fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "n 每加 1，参数表就膨胀一个数量级；语料里大多数 n-gram 只出现一次",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l02-perplexity")
