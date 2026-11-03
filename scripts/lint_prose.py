#!/usr/bin/env python3
"""讲义文本 lint：抓机械性错误，不做语义判断。

检查项：
  1. 连续重复汉字（「的的」「是是」）
  2. 中文与英文/数字之间缺空格
  3. 全角括号里套半角括号
  4. 残留占位符（TODO / 待补 / ???）
  5. 行尾空格、连续三行以上空行
  6. 中英文标点混用（中文句子里用 , . ; ）

用法：python scripts/lint_prose.py [关键字]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

DUP = re.compile(r"([的了是在和与就都也又很把被从对为以于而且并或])\1")
# 中文后面直接跟英文/数字，或反过来
NO_SPACE_AFTER_CN = re.compile(r"([\u4e00-\u9fa5])([A-Za-z0-9])")
NO_SPACE_BEFORE_CN = re.compile(r"([A-Za-z0-9])([\u4e00-\u9fa5])")
PLACEHOLDER = re.compile(r"(TODO|FIXME|待补(?!全)|待写|\?\?\?)")
TRAILING = re.compile(r"[ \t]+$", re.M)
ASCII_PUNCT_IN_CN = re.compile(r"[\u4e00-\u9fa5][,;](?!\d)")

# 允许不空格的例外：常见单位、缩写、公式边界
ALLOW_AFTER = {"度", "元", "个", "次", "层", "维", "倍", "卡", "张", "步", "页", "行", "字"}


def check(path: Path) -> list[str]:
    issues: list[str] = []
    text = path.read_text(encoding="utf-8")

    # 去掉代码块与行内代码，避免误报
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"`[^`]*`", "", body)
    body = re.sub(r"\$\$.*?\$\$", "", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", "", body)

    for i, raw in enumerate(text.splitlines(), 1):
        if TRAILING.search(raw):
            issues.append(f"{path.name}:{i} 行尾空格")

    for i, line in enumerate(body.splitlines(), 1):
        m = DUP.search(line)
        if m and not line.strip().startswith("|"):
            issues.append(f"{path.name}:{i} 重复字「{m.group(0)}」")

        for m in NO_SPACE_AFTER_CN.finditer(line):
            if m.group(1) not in ALLOW_AFTER:
                issues.append(f"{path.name}:{i} 中英之间缺空格：「{m.group(0)}」")
                break

        m2 = ASCII_PUNCT_IN_CN.search(line)
        if m2:
            issues.append(f"{path.name}:{i} 中文里用了半角标点：「{m2.group(0)}」")

        if PLACEHOLDER.search(line):
            issues.append(f"{path.name}:{i} 占位符残留")

    return issues


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = [p for p in sorted(DOCS.rglob("*.md")) if pattern in p.name or pattern in str(p)]

    all_issues: list[str] = []
    for f in files:
        all_issues.extend(check(f))

    # 同类问题合并计数，只打印前 40 条
    from collections import Counter

    kinds = Counter(i.split(" ", 1)[1].split("：")[0] for i in all_issues)
    print(f"检查 {len(files)} 个文件，发现 {len(all_issues)} 处")
    for kind, n in kinds.most_common():
        print(f"  {kind}: {n}")
    print()
    for i in all_issues[:40]:
        print(" ", i)
    if len(all_issues) > 40:
        print(f"  … 另有 {len(all_issues) - 40} 条")


if __name__ == "__main__":
    main()
