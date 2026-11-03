"""L4-3：RoPE 为什么只依赖相对位置。

左图：二维平面上的旋转——位置 m 与 n 的向量各自旋转 mθ、nθ，
      内积只由夹角差 (n-m)θ 决定。
右图：固定相对位移 Δ=5，改变绝对位置 m：
      RoPE 的内积保持不变；而"正弦绝对位置相加"的内积随 m 漂移。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from style import PALETTE, MUTED, clean_axes, save, setup

setup()
rng = np.random.default_rng(7)

# ---------------- 左图：旋转示意 ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax = axes[0]
theta = np.deg2rad(30)
m, n = 1, 4
for pos, color, label in [(m, PALETTE[0], f"位置 $m={m}$"), (n, PALETTE[1], f"位置 $n={n}$")]:
    ang = pos * theta
    ax.annotate(
        "",
        xy=(np.cos(ang), np.sin(ang)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
    )
    ax.text(
        1.12 * np.cos(ang),
        1.12 * np.sin(ang),
        label,
        color=color,
        fontsize=8,
        ha="center",
    )
arc = np.linspace(0, n * theta, 100)
ax.plot(0.35 * np.cos(arc), 0.35 * np.sin(arc), color=MUTED, lw=0.9, ls="--")
ax.text(0.42, 0.12, r"$(n-m)\theta$", color=MUTED, fontsize=8)
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect("equal")
ax.axhline(0, color=MUTED, lw=0.5)
ax.axvline(0, color=MUTED, lw=0.5)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("(a) 旋转把位置写入相位", loc="left")

# ---------------- 右图：内积只依赖 Δ ----------------
d = 64
q = rng.standard_normal(d)
k = rng.standard_normal(d)
omega = 10000 ** (-np.arange(0, d, 2) / d)  # 每对维度的角频率


def rope(vec: np.ndarray, pos: int | float) -> np.ndarray:
    """把 vec 的相邻两维视作一个复数平面并旋转 pos*omega_i。"""
    out = vec.copy()
    ang = pos * omega
    cos, sin = np.cos(ang), np.sin(ang)
    x, y = vec[0::2], vec[1::2]
    out[0::2] = x * cos - y * sin
    out[1::2] = x * sin + y * cos
    return out


def sinusoidal_pe(pos: int | float) -> np.ndarray:
    pe = np.zeros(d)
    pe[0::2] = np.sin(pos * omega)
    pe[1::2] = np.cos(pos * omega)
    return pe


delta = 5
positions = np.arange(0, 60)
rope_dot = np.array([rope(q, float(m)) @ rope(k, float(m + delta)) for m in positions])
sin_dot = np.array([(q + sinusoidal_pe(float(m))) @ (k + sinusoidal_pe(float(m + delta))) for m in positions])

ax = axes[1]
ax.plot(positions, rope_dot, color=PALETTE[0], lw=2, label="RoPE（平移不变）")
ax.plot(positions, sin_dot, color=PALETTE[1], lw=1.6, label="正弦 PE（随 $m$ 漂移）")
ax.set_xlabel(r"绝对位置 $m$（固定 $\Delta=n-m=5$）")
ax.set_ylabel(r"$\langle q_m,\,k_n\rangle$")
ax.set_title("(b) 只有 RoPE 与绝对位置无关", loc="left")
ax.legend(loc="lower right")
ax.set_ylim(min(sin_dot.min(), rope_dot.min()) - 1, max(sin_dot.max(), rope_dot.max()) + 1)
clean_axes(ax)

fig.suptitle("旋转位置编码：把绝对位置编码成相对相位", fontsize=11)
save(fig, "l04-rope")
