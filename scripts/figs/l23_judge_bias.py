"""L23-3：LLM 当裁判的三种偏差。

位置偏差（总偏好第一个）、长度偏差（偏好长答案）、自我偏好（偏好同族模型）。
不校准就拿来评测，结论会系统性偏向某一方。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

biases = ["位置偏差\n（A/B 交换后）", "长度偏差\n（长答案）", "自我偏好\n（同族模型）"]
impact = [0.35, 0.15, 0.10]
colors = [PALETTE[1], PALETTE[3], PALETTE[2]]

fig, ax = plt.subplots(figsize=(6.6, 3.2))
bars = ax.bar(biases, impact, color=colors, width=0.5)
for bar, v in zip(bars, impact):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=8)

ax.set_ylabel("胜率偏移（绝对值）", fontsize=8)
ax.set_title("不校准的裁判，会把结论带偏 10–35 个点", loc="left", fontsize=9.5)
ax.set_ylim(0, 0.45)
clean_axes(ax)

fig.text(0.5, -0.04, "对策：交换顺序取平均、控制长度、混入人类标注做校准",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l23-judge-bias")
