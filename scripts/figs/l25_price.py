"""L25-5：推理价格的三年变化。

纵轴是每百万输入 token 的美元单价（对数），横轴是发布时间。
2023 年的 GPT-4 是 $30，2025 年的 DeepSeek-V3 是 $0.27——两年降了两个数量级，
之后趋于平稳。
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime

from style import INK, MUTED, PALETTE, clean_axes, save, setup

setup()

# (日期, 模型, 每百万输入 token 单价, 颜色)
data = [
    ("2023-03", "GPT-4", 30.0, PALETTE[0]),
    ("2023-11", "GPT-4 Turbo", 10.0, PALETTE[0]),
    ("2024-05", "GPT-4o", 2.5, PALETTE[0]),
    ("2024-12", "DeepSeek-V3", 0.27, PALETTE[1]),
    ("2025-04", "GPT-4.1", 2.0, PALETTE[0]),
    ("2025-08", "GPT-5", 1.25, PALETTE[0]),
    ("2025-12", "DeepSeek-V3.2", 0.28, PALETTE[1]),
    ("2026-03", "GPT-5.4", 2.5, PALETTE[0]),
]

dates = [datetime.strptime(d, "%Y-%m") for d, _, _, _ in data]
x = mdates.date2num(dates)          # 转成 matplotlib 的日期数值，便于类型检查
prices = [p for _, _, p, _ in data]

fig, ax = plt.subplots(figsize=(6.9, 3.4))
ax.plot(x, prices, color=MUTED, lw=1.2, ls="--", zorder=1)

for (d, name, p, color), xi in zip(data, x):
    ax.scatter([xi], [p], s=55, color=color, zorder=5)
    ax.annotate(name, xy=(xi, p), xytext=(0, 8), textcoords="offset points",
                fontsize=6.2, color=color, ha="center")

ax.set_yscale("log")
ax.set_ylim(0.15, 60)
ax.set_ylabel("每百万输入 token 单价（美元，对数）", fontsize=8)
ax.set_xlabel("发布时间", fontsize=8)
ax.set_title("两年降两个数量级，然后趋于平稳", loc="left", fontsize=9.5)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7)

ax.axhspan(0.15, 0.5, color=PALETTE[1], alpha=0.06)
ax.text(x[1], 0.19, "开源/低价区间", fontsize=6.6, color=PALETTE[1])

clean_axes(ax, grid_axis="y")

fig.text(0.5, -0.04, "价格为各模型发布时的官方标价，未考虑缓存、批处理等折扣",
         ha="center", fontsize=7.2, color=INK)

save(fig, "l25-price")
