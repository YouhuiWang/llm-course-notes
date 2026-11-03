"""L17-1：一次完整的 ReAct 轨迹（阶梯图）。

四步"思考 → 动作 → 观测"，最后收敛到答案。
每一步只做一件小事，这正是 §4.4 误差累积公式要求的设计。
"""

from __future__ import annotations

from style import PALETTE, MUTED, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.0, 4.6, xmax=10, ymax=6.4)

# 三列
COL = {
    "think": (0.15, 3.55, PALETTE[0]),
    "act": (3.95, 2.45, PALETTE[1]),
    "obs": (6.70, 3.15, PALETTE[2]),
}
H = 0.78
rows = [5.35, 4.35, 3.35, 2.35]

steps = [
    ("先找所有 Python 文件\n（不确定仓库结构）", 'search_files("**/*.py")', "找到 23 个文件"),
    ("逐个读太慢\n先看目录结构", 'list_dir("src")', "8 个模块，3 个测试"),
    ("统计总行数", 'run_python("wc -l …")', "总计 4821 行"),
    ("核对最大的文件", 'run_python("sort -n …")', "train.py 812 行"),
]

for i, (think, act, obs) in enumerate(steps):
    y = rows[i]
    for key, text, fs in (("think", think, 6.6), ("act", act, 6.4), ("obs", obs, 6.6)):
        x, w, color = COL[key]
        box(ax, x, y, w, H, text, color, fontsize=fs)

    # 行内：思考 → 动作 → 观测
    arrow(
        ax, (COL["think"][0] + COL["think"][1], y + H / 2), (COL["act"][0], y + H / 2), color=MUTED
    )
    arrow(ax, (COL["act"][0] + COL["act"][1], y + H / 2), (COL["obs"][0], y + H / 2), color=MUTED)


# 最终答案
box(
    ax,
    0.15,
    0.75,
    9.55,
    0.9,
    "最终答案：23 个 Python 文件，共 4821 行；最大的是 train.py（812 行）",
    PALETTE[3],
    fontsize=7.4,
    bold=False,
)
arrow(
    ax,
    (COL["obs"][0] + COL["obs"][1] / 2, rows[-1]),
    (COL["obs"][0] + COL["obs"][1] / 2, 1.65),
    color=PALETTE[3],
)

note(
    ax,
    5.0,
    0.30,
    "四步、三次工具调用；每一步都能被单独验证——这是 agent 相对「一次性生成」的核心优势",
    MUTED,
    fontsize=6.8,
)

# 列标题
for key, label in [("think", "思考 Thought"), ("act", "动作 Action"), ("obs", "观测 Observation")]:
    x, w, color = COL[key]
    note(ax, x + w / 2, 6.15, label, color, fontsize=7.2)

save(fig, "l17-trace")
