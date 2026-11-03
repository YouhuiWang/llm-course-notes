"""L25-4：读技术报告的四个问题。

不按论文顺序读，按"能验证什么"读：规模、架构、训练、评测。
每个问题都有对应的"可核对项"，读的时候直接找数字。
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.2, 2.9, xmax=10, ymax=4.2)

items = [
    ("① 规模", "N / D / C 各多少？\n配比是多少？", "算 6ND 对账", PALETTE[0]),
    ("② 架构", "相对标准 Transformer\n改了什么？", "找「我们用了」", PALETTE[2]),
    ("③ 训练", "数据从哪来、怎么清洗？\n怎么防 spike？", "看稳定性和并行", PALETTE[3]),
    ("④ 评测", "有没有污染？\n基线和场景公不公平？", "查污染与基线", PALETTE[1]),
]

W, H = 2.2, 1.7
for i, (name, body, check, color) in enumerate(items):
    x = 0.25 + i * 2.42
    ax.add_patch(FancyBboxPatch((x, 1.45), W, H, boxstyle="round,pad=0.04,rounding_size=0.10",
                                edgecolor=color, facecolor=color, alpha=0.10, lw=1.1))
    ax.text(x + W / 2, 2.85, name, ha="center", fontsize=8.0, color=color)
    ax.text(x + W / 2, 2.20, body, ha="center", fontsize=6.2, color=INK, linespacing=1.5)
    ax.text(x + W / 2, 1.62, check, ha="center", fontsize=5.8, color=MUTED)

note(ax, 5.0, 0.90, "四个问题都能答上来，才算读懂了这份报告", MUTED, fontsize=6.8)
note(ax, 5.0, 0.50, "答不上来的那一项，往往就是报告在回避的地方", PALETTE[1], fontsize=6.6)

save(fig, "l25-reading")
