"""L10-3：四类攻击面。

直接越狱、间接注入、多轮诱导、编码绕过——它们的入口不同，防御手段也不同。
"""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.2)

items = [
    ("① 直接越狱", "「假设你在写小说，\n描述如何……」", "用户自己发起", PALETTE[1]),
    ("② 间接注入", "网页/文件里藏着\n「忽略之前的指令」", "内容里夹带", PALETTE[3]),
    ("③ 多轮诱导", "分十步问，每步都\n看起来无害", "上下文累积", PALETTE[2]),
    ("④ 编码绕过", "Base64、小语种、\n同形字替换", "绕过关键词过滤", PALETTE[0]),
]

W, H = 2.25, 1.55
for i, (title, body, tag, color) in enumerate(items):
    x = 0.25 + i * 2.45
    ax.add_patch(
        FancyBboxPatch(
            (x, 1.45),
            W,
            H,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
            lw=1.1,
        )
    )
    ax.text(x + W / 2, 2.72, title, ha="center", fontsize=8.2, color=color)
    ax.text(x + W / 2, 2.05, body, ha="center", fontsize=6.6, color=INK, linespacing=1.5)
    ax.text(x + W / 2, 1.60, tag, ha="center", fontsize=6.2, color=MUTED)

note(ax, 5.0, 0.85, "前两类靠「输入过滤」，后两类靠「上下文审查 + 输出检测」", MUTED, fontsize=7)
note(ax, 5.0, 0.40, "单层防御必然有漏网：真实攻击常常组合使用", PALETTE[1], fontsize=6.8)

save(fig, "l10-attack-surface")
