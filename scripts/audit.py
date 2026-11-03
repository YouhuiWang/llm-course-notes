#!/usr/bin/env python3
"""按 PLAN 的质量门审计全部已完成讲次。

检查项：
  1. 字数（中文字符，排除代码块 / 公式 / 表格）
  2. 图数、图号是否连续、图注是否含"读图要点"
  3. 算例数（### 例 N.M）
  4. 代码块数
  5. 十段式骨架是否齐全
  6. 每张图是否被正文引用
  7. 公式编号是否连续
  8. 死链

用法：python scripts/audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

SECTIONS = {
    "学习目标": r"^## 学习目标",
    "直觉": r"^## 0 ",
    "形式化": r"^## 1 ",
    "原理": r"^## 2 ",
    "代码": r"^```python",
    "工程": r"^## \d+ .*(工程|要点|坑)",
    "论文": r"^## \d+ .*论文",
    "习题": r"^## \d+ 习题",
    "小结": r"^## \d+ .*小结",
}

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def strip_nonprose(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"<figure.*?</figure>", "", text, flags=re.S)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.M)
    text = re.sub(r"^!\[.*$", "", text, flags=re.M)
    return text


def audit(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    prose = strip_nonprose(text)

    figs = re.findall(r"<figcaption>图 (L\d+-\d+)", text)
    fig_refs = set(re.findall(r"图 (L\d+-\d+)", text))
    examples = re.findall(r"^### 例 (\d+\.\d+)", text, flags=re.M)
    codes = len(re.findall(r"^```python", text, flags=re.M))
    eqs = [int(x) for x in re.findall(r"\\tag\{(\d+)\.(\d+)\}", text) for x in [x[1]]]

    missing = [name for name, pat in SECTIONS.items() if not re.search(pat, text, flags=re.M)]

    # 图号连续性
    nums = [int(f.split("-")[1]) for f in figs]
    fig_ok = nums == list(range(1, len(nums) + 1))

    # 图注质量
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", text, flags=re.S)
    cap_ok = all("读图要点" in c for c in captions)

    # 图是否被正文引用（图注本身不算）
    body = re.sub(r"<figcaption>.*?</figcaption>", "", text, flags=re.S)
    uncited = [f for f in figs if f not in re.findall(r"图 (L\d+-\d+)", body)]

    # 公式编号连续
    eq_ok = eqs == sorted(eqs) and (not eqs or eqs == list(range(1, len(eqs) + 1)))

    # 死链
    broken = []
    for m in LINK_RE.finditer(text):
        t = m.group(1)
        if t.startswith(("http", "mailto", "#", "/", "tel")):
            continue
        if not (path.parent / t.split("#")[0]).exists():
            broken.append(t)

    return {
        "讲": path.stem,
        "字数": len(re.findall(r"[\u4e00-\u9fa5]", prose)),
        "图": len(figs),
        "算例": len(examples),
        "代码": codes,
        "缺段": ",".join(missing) or "—",
        "图号": "✓" if fig_ok else "✗",
        "图注": "✓" if cap_ok else "✗",
        "未引用": len(uncited),
        "公式": "✓" if eq_ok else "✗",
        "死链": len(broken),
    }


def main() -> None:
    files = sorted(DOCS.rglob("l[0-9][0-9]-*.md"))
    rows = [audit(f) for f in files]

    cols = ["讲", "字数", "图", "算例", "代码", "缺段", "图号", "图注", "未引用", "公式", "死链"]
    widths = {c: max(len(str(r[c])) for r in rows + [{c: c}]) for c in cols}
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    bad = [
        r
        for r in rows
        if r["缺段"] != "—"
        or r["图号"] == "✗"
        or r["图注"] == "✗"
        or r["未引用"]
        or r["公式"] == "✗"
        or r["死链"]
    ]
    print()
    print(f"共 {len(rows)} 讲；有问题 {len(bad)} 讲")
    for r in bad:
        print(
            f"  ✗ {r['讲']}: 缺段={r['缺段']} 图号={r['图号']} 图注={r['图注']} "
            f"未引用={r['未引用']} 公式={r['公式']} 死链={r['死链']}"
        )

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
