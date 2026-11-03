"""L21-4：RL 训练中的推理长度增长。

R1-Zero 的一个反直觉现象：没有人教它「要写长」，但随着 RL 推进，
平均回答长度自发增长，准确率同步上升——它在学会「想得更多」。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

steps = np.array([0, 200, 400, 600, 800, 1000, 1500, 2000])
length = np.array([800, 1100, 1800, 2600, 3400, 4200, 5400, 6100])
acc = np.array([0.28, 0.35, 0.44, 0.52, 0.58, 0.63, 0.68, 0.71])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(steps, length, "o-", color=PALETTE[0], lw=2, ms=4, label="平均回答长度")
ax.set_xlabel("RL 训练步数", fontsize=8)
ax.set_ylabel("平均 token 数", fontsize=8, color=PALETTE[0])
ax.tick_params(axis="y", colors=PALETTE[0])

ax2 = ax.twinx()
ax2.plot(steps, acc, "s--", color=PALETTE[2], lw=1.6, ms=4, label="准确率")
ax2.set_ylabel("准确率", fontsize=8, color=PALETTE[2])
ax2.tick_params(axis="y", colors=PALETTE[2])
ax2.spines["top"].set_visible(False)

ax.set_title("没人教它写长，它自己学会了", loc="left", fontsize=9.5)
clean_axes(ax, grid_axis=None)

fig.text(0.5, -0.03,
         "长度涨 7.6 倍，准确率涨 2.5 倍——「想得久」与「想得对」是同时出现的",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l21-length-growth")
