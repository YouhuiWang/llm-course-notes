"""L13-3：量化的显存-精度权衡。

70B 模型：bf16 要 140 GB（两张 80G 卡），INT4 只要 35 GB（一张卡）。
代价是困惑度上升——4 bit 用分组量化能压到 +0.3 以内，3 bit 开始明显掉点。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

bits = np.array([16, 8, 4, 3])
mem = 70e9 * bits / 8 / 1e9
drop = np.array([0.0, 0.05, 0.30, 1.20])

fig, ax = plt.subplots(figsize=(6.6, 3.3))
bars = ax.bar(
    [str(b) for b in bits], mem, color=[PALETTE[2], PALETTE[0], PALETTE[1], PALETTE[3]], width=0.55
)
for bar, m in zip(bars, mem):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 3, f"{m:.0f} GB", ha="center", fontsize=7.5)

ax.axhline(80, color=MUTED, lw=0.9, ls="--")
ax.text(3.42, 83, "单张 80 GB 卡", fontsize=7, color=MUTED, ha="right")

ax.set_xlabel("权重位宽（bit）", fontsize=8)
ax.set_ylabel("70B 模型显存（GB）", fontsize=8)
ax.set_title("量化：用精度换显存", loc="left", fontsize=9.5)
ax.set_ylim(0, 165)
clean_axes(ax)

ax2 = ax.twinx()
ax2.plot(range(4), drop, "o--", color=PALETTE[3], lw=1.4, ms=4)
ax2.set_ylabel("困惑度上升", fontsize=8, color=PALETTE[3])
ax2.tick_params(axis="y", colors=PALETTE[3])
ax2.set_ylim(0, 1.5)
ax2.spines["top"].set_visible(False)

fig.text(
    0.5,
    -0.04,
    "8 bit 几乎无损；4 bit 需分组量化（group=128）；3 bit 以下要谨慎",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l13-quantization")
