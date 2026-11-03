"""L17-2：三层记忆与上下文窗口。

工作记忆在窗口内；情节记忆是外部轨迹日志；语义记忆是持久知识库。
两者通过"写入"与"检索"两条通路与工作记忆交换。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import INK, LIGHT, MUTED, PALETTE, save, setup

setup()

fig, ax = plt.subplots(figsize=(9.4, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, color, title, sub, alpha=0.12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
    )
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=10, color=INK)
    ax.text(x + w / 2, y + h - 0.78, sub, ha="center", va="top", fontsize=7.6, color=MUTED)


def arrow(p0, p1, color, label, rad: float = 0.0, lx: float = 0, ly: float = 0, fs: float = 7.2):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    ax.text(
        (p0[0] + p1[0]) / 2 + lx,
        (p0[1] + p1[1]) / 2 + ly,
        label,
        fontsize=fs,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
    )


# ---- 工作记忆：上下文窗口 ----
box(0.2, 1.0, 2.6, 3.4, PALETTE[0], "工作记忆", "上下文窗口（有限、易失）")
segments = [
    ("系统提示 + 工具描述", 0.62),
    ("最近 3 轮对话", 0.30),
    ("本次工具返回", 0.24),
    ("（剩余空间）", 0.14),
]
y = 3.35
for name, hh in segments:
    ax.add_patch(
        FancyBboxPatch(
            (0.42, y - hh),
            2.16,
            hh - 0.05,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=LIGHT if "剩余" in name else "white",
            edgecolor=MUTED,
            linewidth=0.7,
            linestyle="--" if "剩余" in name else "-",
        )
    )
    ax.text(1.5, y - hh / 2 - 0.02, name, ha="center", va="center", fontsize=6.8, color=INK)
    y -= hh

# ---- 情节记忆 ----
box(3.6, 1.6, 2.5, 2.8, PALETTE[1], "情节记忆", "轨迹日志（可回放、无上限）")
for i, txt in enumerate(["step 1  thought/action/obs", "step 2  thought/action/obs", "step 3  …"]):
    ax.text(3.78, 3.0 - i * 0.36, txt, fontsize=6.4, color=MUTED, family="monospace")

# ---- 语义记忆 ----
box(6.9, 1.6, 2.7, 2.8, PALETTE[2], "语义记忆", "向量库 / 知识图谱（持久）")
for i, txt in enumerate(
    ["· 事实：项目使用 uv 管理", "· 偏好：代码用 4 空格缩进", "· 教训：别直接改 main 分支"]
):
    ax.text(7.05, 3.0 - i * 0.36, txt, fontsize=6.4, color=MUTED)

# ---- 写入 ----
arrow((2.8, 3.5), (3.6, 3.2), PALETTE[1], "写入：每步追加", rad=-0.12, ly=0.30)
arrow((2.8, 2.3), (6.9, 2.2), PALETTE[2], "写入：重要性 $>\\tau$ 才记", rad=0.22, ly=-0.55)

# ---- 检索 ----
arrow((3.6, 1.95), (2.8, 1.75), PALETTE[1], "检索：回放最近轨迹", rad=-0.12, ly=-0.28)
arrow((6.9, 1.9), (2.8, 1.25), PALETTE[2], "检索：recency×importance×relevance", rad=0.16, ly=-0.42)

ax.text(
    5.0,
    0.35,
    "两条通路决定一切：什么值得记（写入），现在该想起什么（检索）",
    ha="center",
    fontsize=8.4,
    color=INK,
)

save(fig, "l17-memory-layers")
