"""L8-3：三种指令数据的构造方式。

人工标注质量最高但最贵；Self-Instruct 用模型生成指令再过滤；
Evol-Instruct 把简单指令"进化"成更难的版本。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 3.2, xmax=10, ymax=4.6)


def row(y, title, steps, color, desc):
    ax.text(0.05, y + 0.30, title, fontsize=7.8, color=color, ha="left")
    x = 0.05
    for i, s in enumerate(steps):
        w = 0.45 + 0.105 * len(s)
        box(ax, x, y - 0.30, w, 0.52, s, color, fontsize=6.4, fill_alpha=0.10)
        if i < len(steps) - 1:
            arrow(ax, (x + w, y - 0.04), (x + w + 0.22, y - 0.04), color=MUTED)
        x += w + 0.22
    note(ax, 9.9, y - 0.04, desc, MUTED, fontsize=6.0, ha="right")


row(4.05, "① 人工标注", ["种子指令", "人工写回答", "质检"], PALETTE[0], "贵，但质量最高")
row(2.85, "② Self-Instruct", ["种子指令", "模型生成", "过滤去重", "扩充"], PALETTE[2], "便宜，需强过滤")
row(1.65, "③ Evol-Instruct", ["简单指令", "改写变难", "生成回答"], PALETTE[3], "提升复杂度")

note(ax, 5.0, 0.55, "三者常混用：人工做种子与质检，模型做规模，进化做难度", MUTED, fontsize=6.8)

save(fig, "l08-data-pipelines")
