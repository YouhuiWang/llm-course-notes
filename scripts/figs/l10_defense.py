"""L10-4：纵深防御。

五道关卡串成一条管道：任何一道被绕过，后面的还能兜住。
图中标出每道关卡"拦得住什么"与"拦不住什么"。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.0, xmax=10, ymax=4.4)

stages = [
    ("输入过滤", "关键词/分类器", "改写、编码绕过"),
    ("系统提示", "角色与边界", "多轮诱导"),
    ("模型对齐", "RLHF 安全数据", "新式越狱"),
    ("输出检测", "有害内容识别", "隐晦表达"),
    ("监控告警", "日志与回溯", "—"),
]

W, H = 1.72, 1.15
for i, (name, how, miss) in enumerate(stages):
    x = 0.22 + i * 1.95
    color = PALETTE[i % len(PALETTE)]
    box(ax, x, 2.05, W, H, "", color, fill_alpha=0.10)
    ax.text(x + W / 2, 2.92, name, ha="center", fontsize=7.6, color=color)
    ax.text(x + W / 2, 2.55, how, ha="center", fontsize=6.2, color=INK)
    ax.text(x + W / 2, 2.20, "漏：" + miss, ha="center", fontsize=5.8, color=MUTED)
    if i < len(stages) - 1:
        arrow(ax, (x + W, 2.62), (x + W + 0.23, 2.62), color=MUTED)

note(ax, 5.0, 1.45, "用户输入 → 五道关卡 → 输出", MUTED, fontsize=7)
note(ax, 5.0, 0.95, "纵深的意义：不指望任何一层完美，只要漏网率相乘后足够低", PALETTE[0], fontsize=6.8)
note(ax, 5.0, 0.50, "若每层漏 10%，五层串联后漏 10⁻⁵——但前提是各层「独立」", PALETTE[1], fontsize=6.8)

save(fig, "l10-defense")
