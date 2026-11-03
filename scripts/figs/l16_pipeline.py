"""L16-1：RAG 的全链路。

上半部分是离线索引（一次性），下半部分是在线查询（每次请求）。
链路上任何一环出错，最终答案都会错——所以评估要分段做。
"""

from __future__ import annotations

from style import INK, MUTED, PALETTE, save, setup
from diagram import arrow, box, canvas, note

setup()

fig, ax = canvas(7.2, 3.4, xmax=10, ymax=4.8)


def chain(y, items, color, label):
    ax.text(0.05, y + 0.52, label, fontsize=7.4, color=color, ha="left")
    x = 0.05
    for i, (name, sub) in enumerate(items):
        w = 1.55
        box(ax, x, y - 0.22, w, 0.62, f"{name}\n{sub}", color, fontsize=6.0)
        if i < len(items) - 1:
            arrow(ax, (x + w, y + 0.09), (x + w + 0.28, y + 0.09), color=MUTED)
        x += w + 0.28


chain(
    3.65,
    [
        ("文档", "PDF/网页/表格"),
        ("切分", "chunking"),
        ("嵌入", "embedding"),
        ("索引", "向量库 + 倒排"),
    ],
    PALETTE[0],
    "离线索引（一次性）",
)

chain(
    2.15,
    [
        ("查询", "用户问题"),
        ("改写", "扩展/路由"),
        ("检索", "混合召回"),
        ("重排", "rerank"),
        ("生成", "带引用"),
    ],
    PALETTE[2],
    "在线查询（每次请求）",
)

note(ax, 5.0, 1.05, "评估要分段：检索错了，生成再好也没用", MUTED, fontsize=7)
note(
    ax,
    5.0,
    0.60,
    "两个最容易被忽略的环节：切分（决定召回上限）与重排（决定精度）",
    PALETTE[1],
    fontsize=6.8,
)

save(fig, "l16-pipeline")
