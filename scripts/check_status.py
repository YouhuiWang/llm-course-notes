#!/usr/bin/env python3
"""校验首页与 README 的进度信息是否一致。

为什么要这个脚本：写作状态在两处维护（docs/index.md 与 README.md），
改一处忘另一处是实际发生过的事故。

检查项：
  1. 两边的里程碑集合相同（M0…M7，不多不少）
  2. 同一里程碑的状态一致（完成 / 进行中 / 待开始）
  3. 两边的"讲次数量"一致（N 讲）
  4. 里程碑不允许写成区间（如 M2–M6），必须是单条

用法：python scripts/check_status.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.md"
README = ROOT / "README.md"

# 形如：| M0 | 骨架… | ✅ 完成 |
ROW = re.compile(r"^\|\s*(M\d(?:[–-]M\d)?)\s*\|([^|]*)\|\s*([^|]*?)\s*\|\s*$", re.M)
COUNT = re.compile(r"(\d+)\s*讲")

STATE_MARK = {"✅": "完成", "🚧": "进行中", "⬜": "待开始"}


def normalize(status: str) -> str:
    for mark, name in STATE_MARK.items():
        if mark in status:
            return name
    if "完成" in status:
        return "完成"
    if "进行" in status:
        return "进行中"
    return "未知"


def parse(path: Path) -> tuple[dict[str, str], set[str]]:
    text = path.read_text(encoding="utf-8")
    rows = ROW.findall(text)
    table: dict[str, str] = {}
    malformed: set[str] = set()
    for milestone, _content, status in rows:
        if "–" in milestone or "-" in milestone:
            malformed.add(milestone)
            continue
        table[milestone] = normalize(status)
    return table, malformed


def lecture_count(path: Path) -> int | None:
    m = COUNT.search(path.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def main() -> None:
    idx, idx_bad = parse(INDEX)
    rdm, rdm_bad = parse(README)

    problems: list[str] = []

    for name, bad in (("docs/index.md", idx_bad), ("README.md", rdm_bad)):
        if bad:
            problems.append(f"{name} 把里程碑写成了区间：{', '.join(sorted(bad))}（应逐条列出）")

    if not idx:
        problems.append("docs/index.md 里没找到「写作状态」表")
    if not rdm:
        problems.append("README.md 里没找到「进度」表")

    only_idx = sorted(set(idx) - set(rdm))
    only_rdm = sorted(set(rdm) - set(idx))
    if only_idx:
        problems.append(f"只在首页出现：{', '.join(only_idx)}")
    if only_rdm:
        problems.append(f"只在 README 出现：{', '.join(only_rdm)}")

    for m in sorted(set(idx) & set(rdm)):
        if idx[m] != rdm[m]:
            problems.append(f"{m} 状态不一致：首页={idx[m]}，README={rdm[m]}")

    ci, cr = lecture_count(INDEX), lecture_count(README)
    if ci and cr and ci != cr:
        problems.append(f"讲次数量不一致：首页={ci} 讲，README={cr} 讲")

    if problems:
        print(f"✗ 进度信息不一致（{len(problems)} 处）：")
        for p in problems:
            print("   ", p)
        sys.exit(1)

    print(f"✓ 进度信息一致：{len(idx)} 个里程碑，{ci} 讲")
    if ci is None or cr is None:
        print("  ⚠ 未能在两处都找到「N 讲」，跳过数量比对")


if __name__ == "__main__":
    main()
