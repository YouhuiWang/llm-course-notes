"""全书统一的绘图样式。

所有插图脚本都必须 `from style import setup, save, PALETTE` 并调用 `setup()`，
保证 24 讲的视觉语言一致：同一套配色、同一套字号、同一套留白。

用法::

    from style import setup, save, clean_axes
    import matplotlib.pyplot as plt

    setup()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ...
    save(fig, "l04-attention-flow")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs" / "assets" / "img"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- 配色：低饱和、色盲友好、黑白打印仍可区分 ----
INK = "#1f2933"  # 正文黑
MUTED = "#7b8794"  # 次要信息
GRID = "#e4e7eb"  # 网格
LIGHT = "#f5f7fa"  # 背景块
PALETTE = [
    "#2f6f9f",  # 蓝
    "#c8553d",  # 砖红
    "#5b8c5a",  # 橄榄绿
    "#b58b00",  # 赭黄
    "#6b5b95",  # 紫
    "#00838f",  # 青
]

CJK_CANDIDATES = [
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
    "Heiti TC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
]


def _pick_cjk() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            return name
    return None


CJK_FONT = _pick_cjk()


def setup() -> None:
    """全局 rcParams。每个脚本开头调用一次。"""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
            "mathtext.fontset": "cm",
            "figure.constrained_layout.use": True,
        }
    )
    if CJK_FONT:
        plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]
    else:  # pragma: no cover - 环境缺失时给出明确警告
        print("⚠ 未找到中文字体，图中中文可能显示为方块；请安装思源黑体或使用英文标注")
        plt.rcParams["font.family"] = ["DejaVu Sans"]


def clean_axes(ax, grid_axis: str | None = "y") -> None:
    """去掉多余边框，按需保留单方向网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name: str) -> Path:
    """同时输出 SVG（网页）与 PDF（印刷），返回 SVG 路径。"""
    svg = FIGDIR / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  ✓ figures/{name}.svg")
    return svg


if __name__ == "__main__":
    setup()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([0, 1, 2], [0, 1, 4], color=PALETTE[0], marker="o")
    ax.set_title("中文字体自检：注意力机制")
    save(fig, "_font_check")
