"""L7-2：上下文学习的 prompt 结构。

任务描述 + k 个示例 + 一个待补全的查询。参数一个都没动，
"学习"全部发生在这一次前向的上下文里。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import box, canvas, note

setup()

fig, ax = canvas(6.9, 3.2, xmax=10, ymax=4.6)

# 三段
box(ax, 0.30, 2.75, 2.5, 1.15, "任务描述\n「把句子翻成英文」", PALETTE[0], fontsize=7.2)
box(ax, 3.05, 2.75, 3.3, 1.15, "$k$ 个示例\n猫 → cat\n狗 → dog", PALETTE[2], fontsize=7.2)
box(ax, 6.60, 2.75, 3.0, 1.15, "查询\n鸟 → ?", PALETTE[1], fontsize=7.2)

# 箭头
from diagram import arrow

arrow(ax, (2.80, 3.32), (3.05, 3.32), color=MUTED)
arrow(ax, (6.35, 3.32), (6.60, 3.32), color=MUTED)

# 下方说明
note(ax, 1.55, 2.15, "系统提示", MUTED, fontsize=6.8)
note(ax, 4.70, 2.15, "示例（可 0 个、1 个或多个）", MUTED, fontsize=6.8)
note(ax, 8.10, 2.15, "待补全", MUTED, fontsize=6.8)

note(ax, 5.0, 1.30,
     "$k=0$：零样本；$k=1$：单样本；$k\\geq 2$：少样本——参数始终冻结",
     PALETTE[0], fontsize=7.2)

note(ax, 5.0, 0.55,
     "整个「学习」发生在这一次前向的上下文里，没有梯度、没有参数更新",
     MUTED, fontsize=6.8)

save(fig, "l07-prompt-structure")
