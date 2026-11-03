"""L7-4：few-shot 的边际收益。

从 0 个示例到 4 个，提升最明显；8 个之后基本饱和。
另一条虚线是"格式对但示例乱序"——说明**格式与顺序**比示例数量更关键。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

k = np.array([0, 1, 2, 4, 8, 16, 32])
task_a = np.array([0.42, 0.61, 0.70, 0.78, 0.82, 0.84, 0.845])
task_b = np.array([0.31, 0.45, 0.55, 0.66, 0.72, 0.75, 0.76])
shuffled = np.array([0.42, 0.52, 0.58, 0.63, 0.66, 0.67, 0.67])

fig, ax = plt.subplots(figsize=(6.6, 3.3))
xs = np.arange(len(k))
ax.plot(xs, task_a, "o-", color=PALETTE[0], lw=1.8, ms=4, label="任务 A（格式正确）")
ax.plot(xs, task_b, "s-", color=PALETTE[2], lw=1.8, ms=4, label="任务 B（格式正确）")
ax.plot(xs, shuffled, "^--", color=PALETTE[1], lw=1.5, ms=4, label="任务 A（示例乱序）")

ax.axvline(3, color=MUTED, lw=0.8, ls=":")
ax.annotate(
    "4 个示例之后\n收益迅速递减",
    xy=(3, 0.78),
    xytext=(3.6, 0.50),
    fontsize=7,
    color=MUTED,
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
)

ax.set_xticks(xs, [str(v) for v in k])
ax.set_xlabel("示例个数 $k$", fontsize=8)
ax.set_ylabel("准确率", fontsize=8)
ax.set_title("few-shot 的收益：数量有限，格式与顺序更关键", loc="left", fontsize=9.5)
ax.set_ylim(0.25, 0.95)
ax.legend(loc="lower right", fontsize=7)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "同样的示例数量，乱序后掉 10–15 个点——ICL 对「格式」极其敏感",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l07-fewshot")
