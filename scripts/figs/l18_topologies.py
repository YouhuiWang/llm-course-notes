"""L18-1：四种多智能体拓扑。

层级（manager-worker）、对等（peer）、辩论（debate）、黑板（blackboard）。
拓扑决定通信成本：全连接是 O(n²)，层级是 O(n)。
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.2, xmax=10, ymax=4.6)


def panel(x0, title, color):
    ax.text(x0 + 1.9, 4.15, title, fontsize=7.6, color=color, ha="center")
    ax.add_patch(
        Rectangle((x0, 0.55), 3.9, 3.35, fill=False, edgecolor=color, lw=0.8, ls="--", alpha=0.5)
    )


panel(0.15, "层级：Manager–Worker", PALETTE[0])
box(ax, 1.55, 3.05, 1.1, 0.5, "Manager", PALETTE[0], fontsize=6.4)
for i in range(3):
    bx = 0.45 + i * 1.2
    box(ax, bx, 1.55, 0.95, 0.45, f"W{i + 1}", PALETTE[2], fontsize=6.2)
    arrow(ax, (2.10, 3.05), (bx + 0.47, 2.00), color=MUTED, lw=0.7, style="<|-|>")

panel(5.10, "对等：全连接", PALETTE[2])
for i in range(3):
    bx = 5.75 + i * 1.25
    box(ax, bx, 2.30, 0.95, 0.45, f"A{i + 1}", PALETTE[2], fontsize=6.2)
for i in range(3):
    for j in range(i + 1, 3):
        xi, xj = 5.75 + i * 1.25 + 0.47, 5.75 + j * 1.25 + 0.47
        arrow(ax, (xi, 2.52), (xj, 2.52), color=MUTED, lw=0.6, style="<|-|>")

note(
    ax, 5.0, 0.20, "左：通信 O(n)，但 Manager 是单点；右：通信 O(n²)，但无单点", MUTED, fontsize=6.8
)

save(fig, "l18-topologies")
