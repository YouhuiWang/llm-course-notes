"""L14-2：ZeRO 三个阶段。

ZeRO-1 只切优化器状态，ZeRO-2 连梯度一起切，ZeRO-3 把参数也切了。
每切一层，单卡显存再降一截，但通信量也涨一截。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

# 7B 模型：bf16 参数 14GB、梯度 14GB、优化器状态 56GB（fp32 master + m + v）
params, grads, opt = 14.0, 14.0, 56.0
world = 8

stages = ["baseline", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
mem = [
    params + grads + opt,
    params + grads + opt / world,
    params + (grads + opt) / world,
    (params + grads + opt) / world,
]

fig, ax = plt.subplots(figsize=(6.6, 3.3))
bars = ax.bar(stages, mem, color=[PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[0]], width=0.55)
for bar, m in zip(bars, mem):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 3, f"{m:.1f} GB", ha="center", fontsize=7.6)

ax.axhline(80, color=MUTED, lw=0.9, ls="--")
ax.text(3.42, 83, "80 GB 卡", fontsize=7, color=MUTED, ha="right")

ax.set_ylabel("单卡显存（GB）", fontsize=8)
ax.set_title("ZeRO：把三份状态逐层切开（8 卡）", loc="left", fontsize=9.5)
ax.set_ylim(0, 130)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "7B 模型：参数 14 GB + 梯度 14 GB + 优化器状态 56 GB = 84 GB",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l14-zero")
