#!/usr/bin/env python3
"""检查文档内部的死链：Markdown 链接、图片、以及 mkdocs.yml 的 nav 条目。

只检查站内相对路径；外链（http/https/mailto）不在此脚本范围内，
需要联网校验时单独跑 `scripts/check_external.py`（M7 阶段实现）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
NAV_RE = re.compile(r"\.md$")


def check_markdown() -> list[str]:
    problems: list[str] = []
    for md in sorted(DOCS.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#", "tel:", "/")):
                continue
            if target.startswith("/"):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            resolved = (md.parent / path).resolve()
            if not resolved.exists():
                problems.append(f"{md.relative_to(ROOT)} → {target}")
    return problems


def check_nav() -> list[str]:
    problems: list[str] = []
    cfg = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and NAV_RE.search(node):
            if not (DOCS / node).exists():
                problems.append(f"mkdocs.yml nav → {node}")

    walk(cfg.get("nav", []))
    return problems


def main() -> None:
    links = check_markdown()
    nav = check_nav()

    if nav:
        print("✗ nav 中有不存在的文件：")
        for p in nav:
            print("   ", p)
    if links:
        print(f"✗ 发现 {len(links)} 个死链：")
        for p in links:
            print("   ", p)

    if not nav and not links:
        total = len(list(DOCS.rglob("*.md")))
        print(f"✓ 死链检查通过（{total} 个 Markdown 文件）")
    sys.exit(1 if (nav or links) else 0)


if __name__ == "__main__":
    main()
