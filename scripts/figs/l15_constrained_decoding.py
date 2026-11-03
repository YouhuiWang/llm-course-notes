"""L15-4：约束解码。

生成 JSON 时，每一步只允许语法上合法的 token：刚写完 `{` 就只能接键名或 `}`，
写完键名就只能接 `:`。非法 token 的概率被置零——这就是"格式保证"的来源。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import canvas, note

setup()

fig, ax = canvas(7.0, 3.0, xmax=10, ymax=4.4)

# 已生成的部分
prefix = ['{', '"name"', ':', '"Ada']
ax.text(0.05, 3.55, "已生成：", fontsize=7.2, color=MUTED)
x = 1.05
for tok in prefix:
    w = 0.42 + 0.1 * len(tok)
    ax.add_patch(Rectangle((x, 3.30), w, 0.42, facecolor=PALETTE[0], alpha=0.14,
                           edgecolor=PALETTE[0], lw=0.9))
    ax.text(x + w / 2, 3.51, tok, ha="center", va="center", fontsize=7, color=INK)
    x += w + 0.08

# 状态
note(ax, 5.0, 2.75, "当前状态：字符串未闭合 —— 只允许字符串内字符或收尾引号",
     PALETTE[2], fontsize=7)

# 候选表
cands = [('"', "✓ 合法", True), ("}", "✗ 屏蔽", False), (",", "✗ 屏蔽", False),
         ("a", "✓ 合法", True), ("\\n", "✗ 屏蔽", False)]
x = 0.55
for tok, label, ok in cands:
    color = PALETTE[2] if ok else PALETTE[1]
    ax.add_patch(Rectangle((x, 1.55), 1.55, 0.7, facecolor=color, alpha=0.12,
                           edgecolor=color, lw=0.9, linestyle="-" if ok else ":"))
    ax.text(x + 0.775, 1.98, f"`{tok}`", ha="center", fontsize=7, color=INK)
    ax.text(x + 0.775, 1.72, label, ha="center", fontsize=6.2, color=color)
    x += 1.75

note(ax, 5.0, 1.05, "把非法候选的 logit 置为 $-\\infty$，再 softmax —— 模型不可能输出半个 JSON",
     MUTED, fontsize=6.8)
note(ax, 5.0, 0.60, "实现：有限状态机（FSM）或下推自动机（PDA）驱动 token 掩码",
     MUTED, fontsize=6.6)
note(ax, 5.0, 0.20, "代价：每步多一次掩码计算；约束太紧会让模型「不会说话」",
     PALETTE[1], fontsize=6.6)

save(fig, "l15-constrained-decoding")
