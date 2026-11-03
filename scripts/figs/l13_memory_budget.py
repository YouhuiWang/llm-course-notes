"""L13-1：显存账本。

7B 模型（bf16）的权重固定 14 GB，但 KV cache 随上下文线性增长。
4K 时还能塞进 24 GB 卡，128K 时连 80 GB 卡都放不下。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

lengths = ["4K", "32K", "128K"]
tokens = np.array([4096, 32768, 131072])
weights = 14.0  # 7B × 2 B
kv = 2 * 32 * 4096 * tokens * 2 / 1e9  # 2·L·d·n·2B

fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(len(lengths))
w = 0.5

b1 = ax.bar(x, [weights] * 3, w, color=PALETTE[0], label="权重（14 GB）")
b2 = ax.bar(x, kv, w, bottom=[weights] * 3, color=PALETTE[1], label="KV cache")

for i, v in enumerate(kv):
    ax.text(i, weights + v + 3, f"{v:.0f} GB", ha="center", fontsize=7.5, color=PALETTE[1])

ax.axhline(24, color=MUTED, lw=0.9, ls="--")
ax.text(2.42, 26, "24 GB 卡", fontsize=7, color=MUTED, ha="right")
ax.axhline(80, color=MUTED, lw=0.9, ls=":")
ax.text(2.42, 82, "80 GB 卡", fontsize=7, color=MUTED, ha="right")

ax.set_xticks(x, lengths)
ax.set_xlabel("上下文长度", fontsize=8)
ax.set_ylabel("显存（GB）", fontsize=8)
ax.set_title("权重是常数，KV cache 是变量", loc="left", fontsize=9.5)
ax.set_ylim(0, 105)
ax.legend(loc="upper left", fontsize=7.2)
clean_axes(ax)

fig.text(
    0.5,
    -0.03,
    "7B 模型、32 层、d=4096、bf16；KV 每 token 512 KB",
    ha="center",
    fontsize=7.2,
    color=INK,
)

save(fig, "l13-memory-budget")
