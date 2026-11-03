r"""L6-4：重复训练的边际收益。

高质量数据不够用时，只能把同一批数据多训几遍。
经验公式：有效数据量 $D_{\text{eff}}=U+R^{*}(1-e^{-R/R^{*}})$，
$R$ 是重复次数，$R^{*}\approx15$ 是"重复到第几遍开始明显不值"。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

R = np.linspace(1, 40, 200)
Rstar = 15.0
eff = Rstar * (1 - np.exp(-R / Rstar))  # 1 遍 ≈ 1 倍，之后逐渐饱和

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(R, R, color=MUTED, lw=0.9, ls=":", label="理想：重复多少遍就多多少")
ax.plot(R, eff, color=PALETTE[0], lw=2, label="实际有效数据量")

for r in (4, 10, 20):
    y = Rstar * (1 - np.exp(-r / Rstar))
    ax.scatter([r], [y], s=24, color=PALETTE[1], zorder=5)
    ax.annotate(
        f"{r}×→{y:.1f}×", xy=(r, y), xytext=(r + 0.8, y - 3.0), fontsize=7, color=PALETTE[1]
    )

ax.set_xlabel("重复遍数 $R$", fontsize=8)
ax.set_ylabel("等效新鲜数据量（倍）", fontsize=8)
ax.set_title("重复训练的收益迅速衰减", loc="left", fontsize=9.5)
ax.legend(loc="upper left", fontsize=7.2)
ax.set_xlim(0, 40)
ax.set_ylim(0, 42)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "重复 4 遍约等于 3.5 倍新数据；重复 20 遍只等于 11 倍——后面每一遍都在打折",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l06-repeat")
