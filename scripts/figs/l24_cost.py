"""L24-2：三种方案的月度成本。

同样的业务量，模型越大、上下文越长，推理成本涨得越快。
这张图决定了"用哪个模型"这个最关键的选型。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

plans = ["小模型 + RAG", "中模型 + RAG", "大模型长上下文"]
infer = np.array([1.2, 4.5, 18.0])
storage = np.array([0.3, 0.3, 0.8])
ops = np.array([0.5, 0.6, 0.9])

x = np.arange(3)
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.bar(x, infer, 0.5, color=PALETTE[0], label="推理")
ax.bar(x, storage, 0.5, bottom=infer, color=PALETTE[2], label="存储/检索")
ax.bar(x, ops, 0.5, bottom=infer + storage, color=PALETTE[1], label="运维/人力")

for i in range(3):
    total = infer[i] + storage[i] + ops[i]
    ax.text(i, total + 0.5, f"{total:.1f} 万", ha="center", fontsize=7.5)

ax.set_xticks(x, plans, fontsize=7)
ax.set_ylabel("月成本（万元）", fontsize=8)
ax.set_title("成本差 10 倍，效果可能只差 3 个点", loc="left", fontsize=9.5)
ax.set_ylim(0, 23)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "选型的关键不是「哪个最好」，而是「哪个够用且付得起」",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l24-cost")
