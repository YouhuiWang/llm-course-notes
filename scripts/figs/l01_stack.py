"""L1-3：大模型应用栈的五层。

从下往上：模型 → 推理 → 工具 → 编排 → 产品。
越往下越接近硬件与训练，越往上越接近用户；本书的内容分布也按这五层展开。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(6.8, 4.0, xmax=10, ymax=6.0)

layers = [
    ("产品层", "对话应用 · IDE 插件 · 客服 · 报表", "用户看到的一切", PALETTE[3]),
    ("编排层", "Agent 循环 · 工作流 · 多智能体", "谁决定下一步做什么", PALETTE[2]),
    ("工具层", "检索（RAG）· 代码执行 · 外部 API", "模型能碰到什么", PALETTE[5]),
    ("推理层", "vLLM · 量化 · KV cache · 投机解码", "一次前向要多久、多少钱", PALETTE[1]),
    ("模型层", "基座预训练 · SFT · 偏好对齐", "权重从哪来", PALETTE[0]),
]

W, H = 9.2, 0.82
for i, (name, examples, question, color) in enumerate(layers):
    y = 4.75 - i * 0.95
    box(ax, 0.35, y, W, H, "", color, fill_alpha=0.10)
    ax.text(
        0.55, y + H / 2, name, ha="left", va="center", fontsize=8.6, color=color, fontweight="bold"
    )
    ax.text(2.35, y + H / 2, examples, ha="left", va="center", fontsize=7.2, color=INK)
    ax.text(9.4, y + H / 2, question, ha="right", va="center", fontsize=6.8, color=MUTED)

arrow(ax, (9.85, 4.2), (9.85, 1.2), color=MUTED, style="-|>")
note(ax, 9.62, 2.7, "越往上越贴近用户", MUTED, fontsize=6.6)
note(ax, 5.0, 0.18, "第 1–14 讲打底下两层，第 15–22 讲覆盖上面三层", MUTED, fontsize=7)

save(fig, "l01-stack")
