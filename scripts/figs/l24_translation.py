"""L24-1：从业务需求到技术指标。

"要快"不是指标，"P95 延迟 < 2 秒"才是。这一讲的第一步就是完成这个翻译。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 2.6, xmax=10, ymax=3.8)

stages = [
    ("业务需求", "客服要快、要准", PALETTE[0]),
    ("可测指标", "P95 < 2s\n准确率 > 85%", PALETTE[2]),
    ("技术选型", "模型 + RAG\n+ 缓存", PALETTE[3]),
    ("验证方法", "离线评测\n+ 灰度", PALETTE[1]),
]

W = 2.0
for i, (name, body, color) in enumerate(stages):
    x = 0.2 + i * 2.42
    box(ax, x, 1.45, W, 1.05, "", color, fill_alpha=0.10)
    ax.text(x + W / 2, 2.22, name, ha="center", fontsize=7.6, color=color)
    ax.text(x + W / 2, 1.78, body, ha="center", fontsize=6.4, color=INK, linespacing=1.5)
    if i < len(stages) - 1:
        arrow(ax, (x + W, 1.97), (x + W + 0.42, 1.97), color=MUTED)

note(ax, 5.0, 0.85, "每一步都要能回溯：指标怎么来的、选型为什么这么定", MUTED, fontsize=6.8)
note(ax, 5.0, 0.45, "最常见的问题：跳过第一步，直接开始调模型", PALETTE[1], fontsize=6.6)

save(fig, "l24-translation")
