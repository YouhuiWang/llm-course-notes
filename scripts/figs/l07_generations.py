"""L7-1：GPT 三代的三条曲线。

参数量、数据量、以及"用不用微调"这条范式线，三者同步跃迁。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

gens = ["GPT-1\n2018", "GPT-2\n2019", "GPT-3\n2020"]
params = [117e6, 1.5e9, 175e9]
data = [5e9, 4e10, 3e11]      # 近似 token 数
paradigm = ["微调", "零样本", "少样本"]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
x = np.arange(3)
w = 0.36

b1 = ax.bar(x - w / 2, params, w, color=PALETTE[0], label="参数量")
b2 = ax.bar(x + w / 2, data, w, color=PALETTE[2], label="训练 token 数")

def fmt(v):
    if v >= 1e10:
        return f"{v/1e9:.0f}B"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    return f"{v/1e6:.0f}M"


for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.25,
                fmt(h), ha="center", fontsize=6.8, color=INK)

for i, p in enumerate(paradigm):
    top = max(params[i], data[i]) * 3.2
    ax.text(i, top, p, ha="center", fontsize=7.6, color=PALETTE[1],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE[1], lw=0.8))

ax.set_yscale("log")
ax.set_xticks(x, gens)
ax.set_ylabel("数量（对数轴）", fontsize=8)
ax.set_title("三代 GPT：规模涨了 1500 倍，范式换了三次", loc="left", fontsize=9.5)
ax.set_ylim(5e7, 1e13)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "参数涨 1500 倍、数据涨 60 倍；真正变的是「怎么用」——微调 → 零样本 → 少样本",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l07-generations")
