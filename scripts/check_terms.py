#!/usr/bin/env python3
"""术语一致性检查。

三件事：
  1. 术语表内部无重复；
  2. "首讲"列引用的讲次文件存在；
  3. 统计术语表中从未在正文出现的条目（提示，不失败）。

术语表是唯一权威：正文首次出现新术语时必须登记。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GLOSSARY = DOCS / "appendix" / "glossary.md"

ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(L\d+)\s*\|", re.M)


def lecture_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for md in DOCS.rglob("l*.md"):
        m = re.match(r"l(\d{2})", md.stem)
        if m:
            files[f"L{int(m.group(1))}"] = md
    return files


def main() -> None:
    text = GLOSSARY.read_text(encoding="utf-8")
    rows = ROW_RE.findall(text)
    terms = [r[0] for r in rows]

    problems: list[str] = []

    dupes = [t for t, c in Counter(terms).items() if c > 1]
    if dupes:
        problems.append(f"术语重复：{', '.join(dupes)}")

    lectures = lecture_files()
    for term, _en, _defn, ref in rows:
        if ref not in lectures:
            problems.append(f"术语「{term}」的首讲 {ref} 不存在")

    # 统计未在正文出现的术语（只提示）
    body = "\n".join(p.read_text(encoding="utf-8") for p in DOCS.rglob("*.md") if p != GLOSSARY)
    missing = [t for t in terms if t not in body]

    if problems:
        print(f"✗ 术语检查发现 {len(problems)} 个问题：")
        for p in problems:
            print("   ", p)
        sys.exit(1)

    print(f"✓ 术语表 {len(terms)} 条，无重复，首讲引用有效")
    if missing:
        print(f"  ℹ {len(missing)} 条尚未在正文出现（正常，随撰写推进减少）")


if __name__ == "__main__":
    main()
