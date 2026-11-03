"""L9-1：RLHF 三阶段流水线（概念图）。

SFT → 奖励模型 → PPO，每一步的输入数据与产出都标出来，
让读者一眼看清"四个模型同时在显存里"是怎么来的。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.4, xmax=10, ymax=5.2)

stages = [
    ("① SFT", "指令微调", "数据：指令–回答对\n产出：$\\pi_{\\text{SFT}}$", PALETTE[0]),
    ("② 奖励模型", "学习人类偏好", "数据：成对比较 $(y_w,y_l)$\n产出：$r_\\phi(x,y)$", PALETTE[1]),
    ("③ PPO", "带 KL 约束的 RL", "数据：提示集 $x$\n产出：$\\pi_\\theta$", PALETTE[2]),
]

W, H = 2.75, 2.55
y0 = 1.55
for i, (title, sub, body, color) in enumerate(stages):
    x = 0.25 + i * 3.35
    box(ax, x, y0, W, H, "", color, fill_alpha=0.10)
    ax.text(x + W / 2, y0 + H - 0.36, title, ha="center", va="center",
            fontsize=9, color=color, fontweight="bold")
    ax.text(x + W / 2, y0 + H - 0.82, sub, ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(x + W / 2, y0 + 0.72, body, ha="center", va="center", fontsize=7.0, color=MUTED, linespacing=1.6)
    if i < 2:
        arrow(ax, (x + W, y0 + H / 2), (x + W + 0.6, y0 + H / 2), color=MUTED)

note(ax, 1.62, 4.62, "只用标注数据", PALETTE[0], fontsize=7)
note(ax, 4.97, 4.62, "需要人类偏好", PALETTE[1], fontsize=7)
note(ax, 8.32, 4.62, "四个模型同显存", PALETTE[2], fontsize=7)

note(ax, 5.0, 0.55,
     "PPO 阶段同时加载：策略 $\\pi_\\theta$、参考 $\\pi_{\\text{ref}}$、奖励 $r_\\phi$、价值 $V$ —— 显存约为 SFT 的 4 倍",
     MUTED, fontsize=6.9)

save(fig, "l09-rlhf-pipeline")
