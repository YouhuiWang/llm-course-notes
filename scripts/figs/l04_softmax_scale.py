"""L4-2：为什么必须除以 sqrt(d_k)。

左图：不同 d_k 下，softmax 最大权重的分布（缩放 vs 不缩放）。
右图：最大权重均值随 d_k 的变化——不缩放时迅速饱和到 1。

结论：不缩放 => softmax 输出接近 one-hot => 梯度消失。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import PALETTE, MUTED, clean_axes, save, setup

setup()
rng = np.random.default_rng(0)

DIMS = [8, 32, 128, 512, 2048]
N_DRAW = 4000


def max_weight(d: int, scale: bool, n: int = N_DRAW) -> np.ndarray:
    """模拟一层注意力：n 个查询 × d 个键，返回每个查询的最大 softmax 权重。

    假设 q、k 的每个分量独立同分布 N(0,1)——这是 LayerNorm 之后激活的
    标准假设，也是 √d_k 论证成立的前提。
    """
    queries = rng.standard_normal((n, d))
    keys = rng.standard_normal((d, d))
    logits = queries @ keys  # (n, d)，方差约为 d
    if scale:
        logits = logits / np.sqrt(d)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

# --- 左：d=512 时两种做法的分布对比 ---
ax = axes[0]
d = 512
for scale, color, label in [
    (False, PALETTE[1], r"不缩放  $q\cdot k$"),
    (True, PALETTE[0], r"缩放  $q\cdot k/\sqrt{d_k}$"),
]:
    w = max_weight(d, scale)
    ax.hist(w, bins=60, density=True, alpha=0.55, color=color, label=label)
ax.set_yscale("log")
ax.set_xlabel("softmax 最大权重")
ax.set_ylabel("密度（对数）")
ax.set_title(f"(a) $d_k={d}$ 时的权重分布", loc="left")
ax.legend(loc="upper center")
clean_axes(ax)

# --- 右：最大权重均值 vs d_k ---
ax = axes[1]
means_unscaled = [max_weight(d, False).mean() for d in DIMS]
means_scaled = [max_weight(d, True).mean() for d in DIMS]
ax.plot(DIMS, means_unscaled, "o-", color=PALETTE[1], label="不缩放")
ax.plot(DIMS, means_scaled, "s-", color=PALETTE[0], label=r"除以 $\sqrt{d_k}$")
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.set_xlabel(r"$d_k$")
ax.set_ylabel("最大权重均值")
ax.set_title("(b) 不缩放时迅速饱和", loc="left")
ax.set_ylim(0, 1.05)
ax.legend(loc="center right")
clean_axes(ax)

fig.suptitle("缩放让 softmax 停留在可训练区间", fontsize=11)
save(fig, "l04-softmax-scale")
