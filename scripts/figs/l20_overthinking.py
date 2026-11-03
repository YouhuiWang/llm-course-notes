"""L20-2：过度思考。

测试时计算不是越多越好。曲线先升后降：超过某个点，模型开始自我怀疑、
反复推翻正确结论，准确率反而下降。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

tokens = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
acc = np.array([0.41, 0.52, 0.63, 0.70, 0.72, 0.70, 0.66, 0.61])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(tokens, acc, "o-", color=PALETTE[0], lw=2, ms=4)
peak = int(np.argmax(acc))
ax.scatter([tokens[peak]], [acc[peak]], s=50, color=PALETTE[2], zorder=5)
ax.annotate(f"最优点 ≈ {tokens[peak]} token", xy=(tokens[peak], acc[peak]),
            xytext=(tokens[peak] * 2.2, acc[peak] + 0.04), fontsize=7.2, color=PALETTE[2],
            arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=0.8))
ax.annotate("过度思考：越算越差", xy=(16384, 0.66), xytext=(2000, 0.50),
            fontsize=7.2, color=PALETTE[1],
            arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.8))

ax.set_xscale("log", base=2)
ax.set_xticks(tokens, [f"{t // 1024}k" if t >= 1024 else str(t) for t in tokens])
ax.set_xlabel("思考 token 数", fontsize=8)
ax.set_ylabel("准确率", fontsize=8)
ax.set_title("想得越多不一定越好", loc="left", fontsize=9.5)
ax.set_ylim(0.35, 0.80)
clean_axes(ax)

fig.text(0.5, -0.03, "原因：额外思考会放大输出方差，把已经正确的答案改错",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l20-overthinking")
