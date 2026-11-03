"""L22-3：持续学习的遗忘问题。

顺序训练任务 A→B→C，A 的性能会持续下滑。
加入少量回放（replay）能显著减缓，但不能完全避免。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

stages = np.arange(0, 4)
plain = np.array([0.90, 0.62, 0.41, 0.28])
replay = np.array([0.90, 0.83, 0.78, 0.74])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(stages, plain, "o-", color=PALETTE[1], lw=2, ms=5, label="顺序训练（无回放）")
ax.plot(stages, replay, "s--", color=PALETTE[0], lw=1.8, ms=5, label="顺序训练 + 10% 回放")

ax.set_xticks(stages, ["训练 A 后", "训练 B 后", "训练 C 后", "训练 D 后"], fontsize=7)
ax.set_ylabel("任务 A 的准确率", fontsize=8)
ax.set_title("灾难性遗忘：学了新的，忘了旧的", loc="left", fontsize=9.5)
ax.set_ylim(0.2, 1.0)
ax.legend(loc="lower left", fontsize=7.2)
clean_axes(ax)

ax.annotate(
    "掉 62 个点",
    xy=(3, 0.28),
    xytext=(1.7, 0.42),
    fontsize=7,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)

fig.text(
    0.5,
    -0.04,
    "回放把遗忘从 62 点压到 16 点——代价是每步多训 10% 旧数据",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l22-forgetting")
