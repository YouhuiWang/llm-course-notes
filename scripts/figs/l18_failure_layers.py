"""L18-3：五层失效模型。

决策、状态、执行、环境、治理——agent 的失败会跨层传播，
定位问题必须先判断"哪一层先坏"。
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 3.2, xmax=10, ymax=4.6)

layers = [
    ("治理", "监控缺失、策略不全、无回滚", PALETTE[3]),
    ("环境", "工具返回噪声、观测延迟、信号错误", PALETTE[5]),
    ("执行", "参数错、调用失败、副作用不可逆", PALETTE[2]),
    ("状态", "记忆过期、上下文丢失、状态不一致", PALETTE[1]),
    ("决策", "目标漂移、计划错误、选错工具", PALETTE[0]),
]

W, H = 8.6, 0.62
for i, (name, fail, color) in enumerate(layers):
    y = 3.55 - i * 0.75
    ax.add_patch(
        FancyBboxPatch(
            (0.55, y),
            W,
            H,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            lw=1.0,
        )
    )
    ax.text(0.75, y + H / 2, name, fontsize=7.6, color=color, va="center")
    ax.text(1.85, y + H / 2, fail, fontsize=6.4, color=INK, va="center")

note(ax, 5.0, 0.42, "失败会向下游传播：决策错了，执行再稳也没用", MUTED, fontsize=6.8)
note(ax, 5.0, 0.12, "排查顺序：先看最上层（决策），再逐层往下", PALETTE[0], fontsize=6.6)

save(fig, "l18-failure-layers")
