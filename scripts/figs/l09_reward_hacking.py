"""L9-4：奖励黑客——代理指标与真实质量的分道扬镳。

优化步数增加时，代理奖励（奖励模型的打分）单调上升；
真实质量先升后降。两条线交叉之后，继续优化只是在"刷分"。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

x = np.linspace(0, 100, 400)
proxy = 1 - np.exp(-x / 40)
true = 1.0 * (1 - np.exp(-x / 18)) - 0.00012 * np.clip(x - 30, 0, None) ** 2

peak = int(np.argmax(true))
# 只在"质量开始下滑之后"找交叉点
tail = np.where(x > 45)[0]
cross = int(tail[np.argmin(np.abs(proxy[tail] - true[tail]))])
cross_x, cross_y = float(x[cross]), float(proxy[cross])

fig, ax = plt.subplots(figsize=(6.8, 3.3))
ax.plot(x, proxy, color=PALETTE[1], lw=2, ls="--", label="代理奖励（奖励模型打分）")
ax.plot(x, true, color=PALETTE[0], lw=2, label="真实质量（人类评审）")

ax.axvspan(cross_x, 100.0, color=PALETTE[1], alpha=0.07)
ax.axvline(cross_x, color=MUTED, lw=0.8, ls=":")
ax.annotate(
    "交叉点之后：分数还在涨，质量在跌",
    xy=(cross_x, cross_y),
    xytext=(cross_x - 30, 0.30),
    fontsize=7.4,
    color=INK,
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
)

peak_x, peak_y = float(x[peak]), float(true[peak])
ax.scatter([peak_x], [peak_y], s=26, color=PALETTE[2], zorder=5)
ax.annotate(
    "早停点",
    xy=(peak_x, peak_y),
    xytext=(peak_x + 3, peak_y + 0.06),
    fontsize=7.4,
    color=PALETTE[2],
)

ax.set_xlabel("优化步数", fontsize=8)
ax.set_ylabel("归一化分值", fontsize=8)
ax.set_title("代理指标的通病：优化到极致 = 刷分", loc="left", fontsize=9.5)
ax.set_ylim(0, 1.12)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "工程做法：在真实质量开始下降前早停，而不是等代理奖励收敛",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l09-reward-hacking")
