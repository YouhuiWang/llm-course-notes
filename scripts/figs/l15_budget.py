"""L15-1：上下文预算。

200K 的窗口不是"能装 200K 就随便装"。系统提示、工具描述、历史、检索片段、
输出预留都要分账，留给"真正有用的内容"的往往不到一半。
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

scenarios = ["对话场景", "RAG 场景"]
parts = {
    "系统提示": [2, 2],
    "工具描述": [8, 8],
    "历史对话": [60, 15],
    "检索片段": [10, 105],
    "当前查询": [1, 1],
    "输出预留": [30, 30],
}
colors = [PALETTE[0], PALETTE[3], PALETTE[2], PALETTE[1], PALETTE[4], MUTED]

fig, ax = plt.subplots(figsize=(7.0, 2.9))
left = [0, 0]
for (name, vals), color in zip(parts.items(), colors):
    bars = ax.barh(scenarios, vals, left=left, color=color, height=0.5, label=name)
    for bar, v, l in zip(bars, vals, left):
        if v >= 8:
            ax.text(l + v / 2, bar.get_y() + bar.get_height() / 2, f"{v}K",
                    ha="center", va="center", fontsize=6.8,
                    color="white" if v > 20 else INK)
    left = [l + v for l, v in zip(left, vals)]

ax.set_xlabel("token 数（K）", fontsize=8)
ax.set_title("200K 窗口的预算分配", loc="left", fontsize=9.5)
ax.set_xlim(0, 210)
ax.legend(ncol=6, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
clean_axes(ax, grid_axis="x")

fig.text(0.5, -0.30, "输出预留必须提前扣掉，否则生成到一半被截断",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l15-budget")
