"""L21-1：推理模型的训练流水线。

冷启动 SFT → RL（RLVR）→ 拒绝采样 → 蒸馏到小模型。
R1-Zero 跳过了第一步，直接纯 RL，反而涌现出反思行为。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 2.9, xmax=10, ymax=4.2)

steps = [
    ("冷启动", "少量长 CoT\n（几千条）", PALETTE[0]),
    ("RL", "可验证奖励\n（对/错）", PALETTE[2]),
    ("拒绝采样", "筛出正确\n轨迹", PALETTE[1]),
    ("蒸馏", "教给小模型", PALETTE[3]),
]

W = 1.85
for i, (name, sub, color) in enumerate(steps):
    x = 0.25 + i * 2.35
    box(ax, x, 2.25, W, 0.9, f"{name}\n{sub}", color, fontsize=6.2)
    if i < len(steps) - 1:
        arrow(ax, (x + W, 2.70), (x + W + 0.5, 2.70), color=MUTED)

# R1-Zero 的旁路
arrow(ax, (1.18, 2.25), (1.18, 1.55), color=PALETTE[1], lw=1.1, dashed=True)
arrow(ax, (1.18, 1.55), (3.62, 1.55), color=PALETTE[1], lw=1.1, dashed=True)
arrow(ax, (3.62, 1.55), (3.62, 2.25), color=PALETTE[1], lw=1.1, dashed=True)
note(ax, 2.40, 1.28, "R1-Zero：跳过冷启动，纯 RL", PALETTE[1], fontsize=6.4)

note(ax, 5.0, 0.85, "四步里最贵的是 RL：每个问题要采多次、跑验证器", MUTED, fontsize=6.8)
note(
    ax,
    5.0,
    0.45,
    "蒸馏是性价比最高的一步：小模型学大模型的推理链，成本低一个数量级",
    PALETTE[3],
    fontsize=6.6,
)

save(fig, "l21-pipeline")
