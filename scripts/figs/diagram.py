"""结构示意图的小工具：统一的方框、箭头与标签。

所有示意图脚本都用同一套坐标约定（0–10 的抽象网格），
避免每张图各自调像素，保证全书风格一致。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import INK, LIGHT, MUTED, PALETTE


def canvas(w: float = 6.8, h: float = 3.4, xmax: float = 10.0, ymax: float = 5.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = PALETTE[0],
    fontsize: float = 8,
    fill_alpha: float = 0.10,
    dashed: bool = False,
    bold: bool = False,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=1.0,
            fill=True,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    # 单独的浅色填充层，避免文字也被 alpha 影响
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            linewidth=0,
            facecolor="white",
            alpha=1 - fill_alpha,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.45,
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def arrow(
    ax,
    p0,
    p1,
    color: str = MUTED,
    rad: float = 0.0,
    lw: float = 1.1,
    style: str = "-|>",
    dashed: bool = False,
    zorder: int = 1,
):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def tag(ax, x: float, y: float, text: str, color: str = MUTED, fontsize: float = 7):
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92),
        zorder=5,
    )


def note(ax, x: float, y: float, text: str, color: str = MUTED, fontsize: float = 7.4, ha: str = "center"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va="center", linespacing=1.4)
