"""L18-4：Agent 的训练数据流。

环境里跑出轨迹 → 过滤与标注 → SFT 学格式 → RL 学策略 → 回到环境验证。
关键难点是"轨迹的信用分配"：一条 20 步的轨迹只有最终对错，不知道哪步错。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 2.9, xmax=10, ymax=4.2)

steps = [
    ("环境", "沙箱/模拟器", PALETTE[4]),
    ("轨迹", "状态-动作-奖励", PALETTE[0]),
    ("过滤", "去重/纠错", PALETTE[2]),
    ("SFT", "学格式与工具用法", PALETTE[1]),
    ("RL", "学策略（稀疏奖励）", PALETTE[3]),
]

W = 1.55
for i, (name, sub, color) in enumerate(steps):
    x = 0.25 + i * 1.95
    box(ax, x, 2.15, W, 0.85, f"{name}\n{sub}", color, fontsize=5.9)
    if i < len(steps) - 1:
        arrow(ax, (x + W, 2.57), (x + W + 0.4, 2.57), color=MUTED)

# 回到环境的闭环
arrow(ax, (9.0, 2.15), (9.0, 1.35), color=PALETTE[3], lw=1.2)
arrow(ax, (9.0, 1.35), (1.0, 1.35), color=PALETTE[3], lw=1.2)
arrow(ax, (1.0, 1.35), (1.0, 2.15), color=PALETTE[3], lw=1.2)
note(ax, 5.0, 1.12, "回到环境验证：训练与部署用同一套评测", PALETTE[3], fontsize=6.6)

note(ax, 5.0, 0.55, "难点：一条 20 步轨迹只有最终对错——信用分配（credit assignment）", MUTED, fontsize=6.8)
note(ax, 5.0, 0.20, "对策：过程奖励模型、拒绝采样、把长任务拆成可验证的子目标",
     PALETTE[1], fontsize=6.6)

save(fig, "l18-training")
