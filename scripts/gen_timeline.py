#!/usr/bin/env python3
"""从 refs/timeline.yaml 生成 docs/appendix/timeline.md。

数据驱动：改 YAML 即可改图与表，避免正文与数据两处维护。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "refs" / "timeline.yaml"
DST = ROOT / "docs" / "appendix" / "timeline.md"

CATEGORY_ICON = {
    "理论": "🧠",
    "架构": "🏗️",
    "规模": "📈",
    "对齐": "🎯",
    "应用": "🚀",
    "开源": "🔓",
    "系统": "⚙️",
    "政策": "⚖️",
}


def decade(date: str) -> str:
    year = int(date[:4])
    return f"{year // 10 * 10}s"


def main() -> None:
    events = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        groups[decade(e["date"])].append(e)

    lines = [
        "# 大模型发展时间线",
        "",
        "> 本页由 `refs/timeline.yaml` 自动生成，请勿手工编辑。",
        "> 运行 `python scripts/gen_timeline.py` 更新。",
        "",
        f"共收录 **{len(events)}** 个节点，覆盖 {min(int(e['date'][:4]) for e in events)}–"
        f"{max(int(e['date'][:4]) for e in events)} 年。",
        "",
    ]

    for dec in sorted(groups):
        lines.append(f"## {dec}")
        lines.append("")
        lines.append("| 时间 | 事件 | 类别 | 一句话 |")
        lines.append("|---|---|---|---|")
        for e in sorted(groups[dec], key=lambda x: x["date"]):
            icon = CATEGORY_ICON.get(e["category"], "•")
            flag = "" if e.get("verified", True) else " ⚠️"
            title = f"[{e['title']}]({e['link']})" if e.get("link") else e["title"]
            lines.append(
                f"| {e['date']}{flag} | {title}<br><small>{e.get('en', '')}</small> "
                f"| {icon} {e['category']} | {e['summary']} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 读时间线的三个视角",
        "",
        "1. **架构线**：RNN（1997）→ Attention（2014）→ Transformer（2017）→ MoE（2020/2024）→ 长上下文（2024）",
        "2. **规模线**：NNLM（2003）→ GPT-3（2020）→ Scaling Laws（2020）→ Chinchilla（2022）→ 推理期扩展（2024）",
        "3. **对齐线**：InstructGPT（2022）→ DPO（2023）→ o1/R1（2024–2025）",
        "",
        "三条线的交汇点，就是「能用的助手」出现的位置。",
        "",
    ]

    DST.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已生成 {DST.relative_to(ROOT)}（{len(events)} 条）")


if __name__ == "__main__":
    main()
