"""L5-4：三种激活函数。

ReLU 简单但不可导点硬；GELU 平滑；SiLU（Swish）在负区仍有小输出，
是 SwiGLU 的门控基础。右侧标出各代模型的选择。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

x = np.linspace(-4, 4, 400)
relu = np.maximum(0, x)
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
silu = x / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(x, relu, color=PALETTE[1], lw=1.6, label="ReLU（GPT-1）")
ax.plot(x, gelu, color=PALETTE[0], lw=1.8, label="GELU（GPT-2/BERT）")
ax.plot(x, silu, color=PALETTE[2], lw=1.8, label="SiLU / Swish（LLaMA）")

ax.axhline(0, color=MUTED, lw=0.6)
ax.axvline(0, color=MUTED, lw=0.6)

ax.annotate(
    "负区仍有小输出\n→ 梯度不会完全断",
    xy=(-1.6, -1.6 / (1 + np.exp(1.6))),
    xytext=(-3.8, -1.15),
    fontsize=6.8,
    color=PALETTE[2],
    arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=0.8),
)

ax.set_xlabel("$x$", fontsize=8)
ax.set_ylabel("激活值", fontsize=8)
ax.set_title("从 ReLU 到 SiLU：平滑性越来越重要", loc="left", fontsize=9.5)
ax.legend(loc="upper left", fontsize=7)
clean_axes(ax)

fig.text(
    0.5, -0.03,
    "SwiGLU = SiLU(门控分支) ⊙ 线性分支，再降维——三个矩阵换更好的表达力",
    ha="center", fontsize=7.2, color=INK,
)

save(fig, "l05-activations")
