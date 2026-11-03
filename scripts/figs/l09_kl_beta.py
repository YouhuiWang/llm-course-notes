r"""L9-3：β 控制策略能偏离参考模型多远。

两个动作的玩具问题：参考策略各 0.5，奖励 $r=(1,0)$。
最优策略 $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$，β 越小越激进，
但代价是 KL 越大——这就是"奖励"与"别跑偏"的拉扯。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

beta = np.logspace(-1.3, 0.7, 200)
pi1 = 1 / (1 + np.exp(-1 / beta))
pi2 = 1 - pi1
kl = pi1 * np.log(pi1 / 0.5) + pi2 * np.log(pi2 / 0.5)

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(beta, pi1, color=PALETTE[0], lw=2, label=r"$\pi^\star(a_1)$")
ax.set_xscale("log")
ax.set_xlabel(r"KL 系数 $\beta$（对数轴）", fontsize=8)
ax.set_ylabel("最优策略给好动作的概率", fontsize=8, color=PALETTE[0])
ax.set_ylim(0.45, 1.02)
ax.tick_params(axis="y", colors=PALETTE[0])

ax2 = ax.twinx()
ax2.plot(beta, kl, color=PALETTE[1], lw=1.6, ls="--", label=r"$D_{\mathrm{KL}}(\pi^\star\|\pi_{\mathrm{ref}})$")
ax2.set_ylabel("KL 散度", fontsize=8, color=PALETTE[1])
ax2.tick_params(axis="y", colors=PALETTE[1])
ax2.spines["top"].set_visible(False)

for b, note_txt in [(1.0, r"$\beta=1$"), (0.1, r"$\beta=0.1$")]:
    v = 1 / (1 + np.exp(-1 / b))
    ax.scatter([b], [v], s=26, color=PALETTE[3], zorder=5)
    ax.annotate(note_txt, xy=(b, v), xytext=(b * 0.82, v - 0.07), fontsize=7.4, color=PALETTE[3])

ax.set_title("β 越小，越敢偏离参考模型", loc="left", fontsize=9.5)
handles = [ax.get_lines()[0], ax2.get_lines()[0]]
labels: list[str] = [r"$\pi^\star(a_1)$", r"$D_{\mathrm{KL}}(\pi^\star\|\pi_{\mathrm{ref}})$"]
ax.legend(handles, labels, loc="lower left", fontsize=7.2)
clean_axes(ax, grid_axis=None)

fig.text(
    0.5, -0.03,
    r"β=1 时 KL 仅 0.11；β=0.1 时 KL 逼近上限 ln2≈0.69——模型几乎完全倒向好动作",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l09-kl-beta")
