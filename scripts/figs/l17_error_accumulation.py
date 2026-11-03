"""L17-2：长程任务里的误差累积。

Agent 的每一步都有失败概率。即使单步成功率高达 95%，
20 步之后整体成功率也只有 0.95^20 ≈ 36%。

左图：整体成功率随步数衰减。
右图：要达到 50% 的整体成功率，允许的最长链路长度。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import PALETTE, MUTED, clean_axes, save, setup

setup()

steps = np.arange(1, 31)
probs = [0.99, 0.95, 0.90, 0.80]
colors = PALETTE[:4]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

ax = axes[0]
for p, color in zip(probs, colors):
    ax.plot(steps, p**steps, color=color, lw=1.8, label=f"单步 $p={p:.2f}$")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.annotate(
    r"$0.95^{20}\approx36\%$",
    xy=(20, 0.95**20),
    xytext=(21.5, 0.52),
    fontsize=8,
    color=PALETTE[1],
    arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8),
)
ax.set_xlabel("任务步数 $n$")
ax.set_ylabel("整体成功率 $p^n$")
ax.set_title("(a) 单步可靠 ≠ 任务可靠", loc="left")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", ncol=2)
clean_axes(ax)

ax = axes[1]
target = 0.5
horizons = [np.log(target) / np.log(p) for p in probs]
bars = ax.barh([f"$p={p:.2f}$" for p in probs], horizons, color=colors, height=0.55)
for bar, h in zip(bars, horizons):
    ax.text(h + 0.4, bar.get_y() + bar.get_height() / 2, f"{h:.0f} 步", va="center", fontsize=8)
ax.axvline(20, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("整体成功率跌到 50% 所需的步数")
ax.set_title("(b) 50% 成功率对应的任务长度上限", loc="left")
ax.set_xlim(0, max(horizons) * 1.15)
clean_axes(ax, grid_axis="x")

fig.suptitle("为什么 agent 要缩短链路、加检查点", fontsize=11)
save(fig, "l17-error-accumulation")
