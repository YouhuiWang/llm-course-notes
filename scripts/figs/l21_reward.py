"""L21-2：RLVR 的奖励构成。

可验证奖励 = 正确性 + 格式 + 长度惩罚。正确性占大头，
格式奖励防止输出无法解析，长度惩罚防止无限啰嗦。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

cases = ["答对且格式对\n（简洁）", "答对但啰嗦\n（超长）", "答错但格式对", "答错且格式错"]
correct = np.array([1.0, 1.0, 0.0, 0.0])
format_b = np.array([0.2, 0.2, 0.2, 0.0])
penalty = np.array([0.0, -0.15, 0.0, -0.1])

x = np.arange(4)
fig, ax = plt.subplots(figsize=(6.8, 3.2))
ax.bar(x, correct, 0.5, color=PALETTE[0], label="正确性（+1.0）")
ax.bar(x, format_b, 0.5, bottom=correct, color=PALETTE[2], label="格式（+0.2）")
ax.bar(x, penalty, 0.5, color=PALETTE[1], label="长度惩罚")

for i in range(4):
    total = correct[i] + format_b[i] + penalty[i]
    ax.text(i, total + 0.05, f"{total:.2f}", ha="center", fontsize=7.5, color=INK)

ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x, cases, fontsize=6.6)
ax.set_ylabel("奖励", fontsize=8)
ax.set_ylim(-0.3, 1.4)
ax.set_title("奖励不是单一数字，而是几项的加权和", loc="left", fontsize=9.5)
ax.legend(loc="upper right", fontsize=6.8)
clean_axes(ax)

fig.text(0.5, -0.05, "正确性必须占绝对主导，否则模型会学会「格式漂亮但答错」",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l21-reward")
