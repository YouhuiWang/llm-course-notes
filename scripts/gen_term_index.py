#!/usr/bin/env python3
"""从 glossary.md 生成「按讲次分组」的术语索引。

术语表是单一数据源：每条带「首讲」列。本脚本按讲次聚合，
生成 docs/appendix/term-index.md，供读者按进度查阅。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "docs" / "appendix" / "glossary.md"
OUT = ROOT / "docs" / "appendix" / "term-index.md"

ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(L\d+)\s*\|", re.M)


def main() -> None:
    text = GLOSSARY.read_text(encoding="utf-8")
    rows = ROW.findall(text)

    by_lecture: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for term, en, definition, ref in rows:
        by_lecture[ref].append((term, en, definition))

    lines = [
        "# 术语索引（按讲次）",
        "",
        "> 本页由 `scripts/gen_term_index.py` 从[术语表](glossary.md)生成，请勿手工编辑。",
        "> 用途：按学习进度回查——读到第 N 讲时，本讲引入了哪些新词。",
        "",
        f"共 **{len(rows)}** 条术语，分布在 {len(by_lecture)} 个讲次。",
        "",
    ]

    for ref in sorted(by_lecture, key=lambda x: int(x[1:])):
        items = sorted(by_lecture[ref], key=lambda t: t[0])
        lines.append(f"## 第 {ref[1:]} 讲")
        lines.append("")
        lines.append("| 术语 | English | 定义 |")
        lines.append("|---|---|---|")
        for term, en, definition in items:
            lines.append(f"| {term} | {en} | {definition} |")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已生成 {OUT.relative_to(ROOT)}（{len(rows)} 条，{len(by_lecture)} 个讲次）")


if __name__ == "__main__":
    main()
