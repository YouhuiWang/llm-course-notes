"""L7-3：上下文学习的三种解释。

参数没变，变的是"模型认为自己处在什么任务里"。三条假说分别从
检索、贝叶斯推断、隐式优化的角度解释这件事。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import box, canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.2)

note(ax, 5.0, 3.85, "参数冻结，变的是「模型认为自己处在什么任务里」", INK, fontsize=8)

items = [
    ("① 任务定位", "预训练见过这个任务，\nprompt 把它「找出来」", PALETTE[0]),
    ("② 隐式贝叶斯", "prompt 是证据，\n更新对潜在任务的信念", PALETTE[2]),
    ("③ 隐式梯度下降", "前向计算等价于\n在上下文上走了几步梯度", PALETTE[3]),
]

W, H = 2.85, 1.75
for i, (title, body, color) in enumerate(items):
    x = 0.35 + i * 3.25
    box(ax, x, 1.55, W, H, "", color, fill_alpha=0.10)
    ax.text(x + W / 2, 2.95, title, ha="center", va="center", fontsize=8.4, color=color)
    ax.text(
        x + W / 2, 2.15, body, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.6
    )

note(
    ax,
    5.0,
    0.75,
    "三者不互斥：检索解释「为什么能」，贝叶斯解释「怎么权衡」，梯度下降解释「为何几步就够」",
    MUTED,
    fontsize=6.8,
)
note(
    ax,
    5.0,
    0.35,
    "共同的反证：ICL 对示例顺序、标签是否正确都敏感——说明它不是真正的「学习」",
    PALETTE[1],
    fontsize=6.6,
)

save(fig, "l07-icl-explanations")
