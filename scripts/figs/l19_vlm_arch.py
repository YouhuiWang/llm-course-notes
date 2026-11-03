"""L19-3：三种视觉-语言架构。

(a) projector：图像编码后经 MLP 映射成若干 token，与文本拼在一起；
(b) cross-attention：图像特征在每一层通过交叉注意力注入；
(c) native：图文从一开始就共用同一个 tokenizer 与 Transformer。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.0, xmax=10, ymax=4.4)

panels = [
    ("(a) Projector（LLaVA）", PALETTE[0], ["图像", "ViT", "MLP", "拼接 → LLM"]),
    ("(b) 交叉注意力（Flamingo）", PALETTE[2], ["图像", "ViT", "每层 cross-attn"]),
    ("(c) 原生多模态", PALETTE[3], ["图像/文本", "统一 tokenizer", "单一 Transformer"]),
]

for i, (title, color, steps) in enumerate(panels):
    y = 3.15 - i * 1.15
    ax.text(0.05, y + 0.45, title, fontsize=7.2, color=color, ha="left")
    x = 2.55
    for j, s in enumerate(steps):
        w = 0.55 + 0.115 * len(s)
        box(ax, x, y - 0.05, w, 0.5, s, color, fontsize=6.0, fill_alpha=0.10)
        if j < len(steps) - 1:
            arrow(ax, (x + w, y + 0.2), (x + w + 0.2, y + 0.2), color=MUTED)
        x += w + 0.2

note(ax, 5.0, 0.55, "共同点：都把图像变成某种「token」；差别在「在哪一层融合」", MUTED, fontsize=6.8)
note(ax, 5.0, 0.20, "融合越早，模型越统一，但训练成本越高", PALETTE[1], fontsize=6.6)

save(fig, "l19-vlm-arch")
