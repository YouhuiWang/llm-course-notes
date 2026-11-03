"""L17-6：记忆检索的加权打分。

五条候选记忆，三个分量各自归一化到 [0,1]，
最终得分 = 0.3·recency + 0.3·importance + 0.4·relevance。
最"新"的那条并不一定排第一——这正是加权的意义。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

items = [
    ("临时路径 /tmp/x", 0.99, 0.2, 0.2),
    ("上次改了 config.py", 0.95, 0.5, 0.3),
    ("教训：别直接改 main", 0.60, 0.9, 0.4),
    ("项目用 uv 管依赖", 0.80, 0.7, 0.8),
    ("用户偏好 4 空格缩进", 0.90, 0.8, 0.7),
]
names = [i[0] for i in items]
rec = np.array([i[1] for i in items])
imp = np.array([i[2] for i in items])
rel = np.array([i[3] for i in items])
score = 0.3 * rec + 0.3 * imp + 0.4 * rel

order = np.argsort(score)
names = [names[i] for i in order]
rec, imp, rel, score = rec[order], imp[order], rel[order], score[order]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(names))
h = 0.26

ax.barh(y + h, rec, height=h, color=PALETTE[0], label="recency ×0.3")
ax.barh(y, imp, height=h, color=PALETTE[1], label="importance ×0.3")
ax.barh(y - h, rel, height=h, color=PALETTE[2], label="relevance ×0.4")
ax.scatter(score, y, marker="D", s=28, color=PALETTE[3], zorder=5, label="加权总分")

for yi, s in zip(y, score):
    ax.text(s + 0.02, yi + 0.02, f"{s:.2f}", fontsize=7, color=PALETTE[3], va="center")

ax.set_yticks(y, names, fontsize=7.6)
ax.set_xlim(0, 1.12)
ax.set_xlabel("归一化分值", fontsize=8)
ax.set_title("检索时，最新的不一定最相关", loc="left", fontsize=9.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.8, ncol=4, handlelength=1.2, columnspacing=1.2)
clean_axes(ax, grid_axis="x")


save(fig, "l17-memory-scores")
