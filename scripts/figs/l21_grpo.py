"""L21-3：GRPO 的组内归一化。

同一道题采 8 次，用组内均值与标准差归一化出优势，
不需要额外的价值网络——这是 GRPO 比 PPO 省显存的原因。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

rewards = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=float)
mean = rewards.mean()
std = rewards.std()
adv = (rewards - mean) / (std + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw={"width_ratios": [1, 1]})

ax = axes[0]
colors = [PALETTE[2] if r > 0 else PALETTE[1] for r in rewards]
ax.bar(range(8), rewards, color=colors, width=0.6)
ax.axhline(mean, color=MUTED, lw=1.0, ls="--")
ax.text(7.4, mean + 0.04, f"均值 {mean:.3f}", fontsize=6.8, color=MUTED, ha="right")
ax.set_xticks(range(8), [f"s{i + 1}" for i in range(8)], fontsize=6.6)
ax.set_ylabel("奖励（对=1，错=0）", fontsize=7.5)
ax.set_title("(a) 同一题采 8 次", loc="left", fontsize=9)
ax.set_ylim(-0.1, 1.2)
clean_axes(ax)

ax = axes[1]
ax.bar(range(8), adv, color=[PALETTE[2] if a > 0 else PALETTE[1] for a in adv], width=0.6)
ax.axhline(0, color=MUTED, lw=0.8)
for i, a in enumerate(adv):
    ax.text(i, a + (0.08 if a > 0 else -0.14), f"{a:+.2f}", ha="center", fontsize=6.2)
ax.set_xticks(range(8), [f"s{i + 1}" for i in range(8)], fontsize=6.6)
ax.set_ylabel("优势（归一化后）", fontsize=7.5)
ax.set_title("(b) 组内归一化得到优势", loc="left", fontsize=9)
ax.set_ylim(-1.4, 1.8)
clean_axes(ax)

fig.text(0.5, -0.06,
         f"均值 {mean:.3f}、标准差 {std:.3f}；优势 = (奖励 − 均值) / 标准差",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l21-grpo")
