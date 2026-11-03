r"""L6-3：三代模型的"每参数配多少数据"。

GPT-3 约 1.7 token/参数（严重训练不足）；Chinchilla 用 20；LLaMA 直接把比例拉到 150 以上。
这张图一眼就能看出"堆参数"和"喂数据"两条路线的分歧。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

models = [
    ("GPT-3\n175B / 300B", 175e9, 300e9, PALETTE[1]),
    ("Chinchilla\n70B / 1.4T", 70e9, 1.4e12, PALETTE[2]),
    ("LLaMA-7B\n6.7B / 1T", 6.7e9, 1e12, PALETTE[0]),
    ("LLaMA-65B\n65B / 1.4T", 65e9, 1.4e12, PALETTE[3]),
]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
names = [m[0] for m in models]
ratios = [m[2] / m[1] for m in models]
colors = [m[3] for m in models]

bars = ax.bar(names, ratios, color=colors, width=0.55)
for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, r * 1.08, f"{r:.0f}", ha="center",
            fontsize=8, color=INK)

ax.axhline(20, color=MUTED, lw=0.9, ls="--")
ax.text(3.42, 22, "Chinchilla 推荐 20", fontsize=7, color=MUTED, ha="right")

ax.set_yscale("log")
ax.set_ylabel("训练 token 数 / 参数量", fontsize=8)
ax.set_title("同一笔算力，喂多少数据才划算", loc="left", fontsize=9.5)
ax.set_ylim(0.8, 400)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "GPT-3 只喂了 1.7 token/参数，属于「大模型饿着训」；LLaMA 反其道而行",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l06-token-ratio")
