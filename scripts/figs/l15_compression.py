"""L15-3：三种上下文压缩策略。

摘要：把旧历史压成几句话；裁剪：只保留与当前问题相关的片段；
卸载：把内容存到外部，需要时再检索回来。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.4)

rows = [
    ("① 摘要", ["20 轮历史", "摘要成 5 句", "继续对话"], PALETTE[0], "省 80% token，丢细节"),
    ("② 裁剪", ["100 段检索", "按相关度排序", "保留前 8 段"], PALETTE[2], "需要可靠的重排模型"),
    ("③ 卸载", ["长文档", "存入向量库", "按需检索回"], PALETTE[3], "本质是 RAG（第 16 讲）"),
]

for i, (name, steps, color, desc) in enumerate(rows):
    y = 3.35 - i * 1.05
    ax.text(0.05, y + 0.22, name, fontsize=7.8, color=color, ha="left")
    x = 1.15
    for j, s in enumerate(steps):
        w = 0.5 + 0.115 * len(s)
        box(ax, x, y - 0.18, w, 0.46, s, color, fontsize=6.4, fill_alpha=0.10)
        if j < len(steps) - 1:
            arrow(ax, (x + w, y + 0.05), (x + w + 0.2, y + 0.05), color=MUTED)
        x += w + 0.2
    note(ax, 9.95, y + 0.05, desc, MUTED, fontsize=6.0, ha="right")

note(
    ax, 5.0, 0.45, "三者常组合：先用摘要压历史，再用裁剪压检索，超长文档走卸载", MUTED, fontsize=6.8
)

save(fig, "l15-compression")
