#!/usr/bin/env python3
"""把讲义导出成普通 Markdown 笔记，供 Obsidian / Typora / VS Code 直接阅读。

与站点的差别：
  - 去掉 MkDocs 专有语法（admonition、figure 的 markdown="span"、attr_list）
  - 图片路径改成本地 images/，并复制插图
  - 章间链接改成目标文件名
  - 保留 $$ 公式、表格、代码块

用法：
    python scripts/export_notes.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = Path("/Users/wyh/Desktop/Java 后端开发 + AI 应用/笔记/大模型")

LECTURE_ORDER = [
    "00-导论/l01-what-is-llm",
    "01-前史/l02-statistical-lm",
    "01-前史/l03-neural-seq",
    "02-transformer/l04-attention",
    "02-transformer/l05-block-and-paradigms",
    "02-transformer/l06-scaling-laws",
    "02-transformer/l07-gpt-lineage",
    "03-对齐/l08-sft-data",
    "03-对齐/l09-preference-optimization",
    "03-对齐/l10-safety-redteam",
    "04-架构与系统/l11-moe",
    "04-架构与系统/l12-long-context",
    "04-架构与系统/l13-inference-systems",
    "04-架构与系统/l14-training-systems",
    "05-应用层/l15-context-engineering",
    "05-应用层/l16-rag",
    "05-应用层/l17-agent-1",
    "05-应用层/l18-agent-2",
    "05-应用层/l19-multimodal",
    "06-推理与前沿/l20-test-time-compute",
    "06-推理与前沿/l21-reasoning-training",
    "06-推理与前沿/l22-frontier",
    "07-治理与实践/l23-evaluation-governance",
    "07-治理与实践/l24-capstone",
    "08-模型解剖/l25-model-anatomy",
]

LABS = [
    "lab1-attention", "lab2-minigpt", "lab3-bpe", "lab4-lora",
    "lab5-inference", "lab6-rag", "lab7-agent", "lab8-test-time",
]

APPENDIX = [
    "glossary", "term-index", "papers", "timeline",
    "answers", "faq", "math", "env", "report-template",
]

ADMONITION = re.compile(r'^!!!\s+(\w+)(?:\s+"(.*)")?\s*\n((?:^(?:    |\t).*\n|\n)*)', re.M)
FIGURE = re.compile(r"<figure markdown=\"span\">\s*\n(.*?)\n</figure>", re.DOTALL)
IMG_ATTR = re.compile(r"\{[^}]*\}")
LINK = re.compile(r"\]\(([^)\s]+?\.md)(#[^)]*)?\)")


def slug(text: str) -> str:
    """取 H1 里「第 N 讲　标题」的标题部分"""
    text = re.sub(r"\{[^}]*\}", "", text.strip())
    text = re.sub(r"^#\s*", "", text.strip())
    text = re.sub(r"^第\s*\d+\s*讲[\s　]+", "", text)
    text = re.sub(r"^Lab\s*\d+[\s　]+", "", text)
    return text.strip()


def dedent_block(block: str) -> str:
    lines = []
    for ln in block.splitlines():
        if ln.startswith("    "):
            lines.append(ln[4:])
        elif ln.startswith("\t"):
            lines.append(ln[1:])
        else:
            lines.append(ln)
    return "\n".join(lines).rstrip()


def convert_admonitions(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        kind, title, body = m.group(1), m.group(2), m.group(3)
        body = dedent_block(body).strip()
        head = title or {"note": "说明", "tip": "要点", "warning": "注意", "danger": "警告",
                         "abstract": "本讲位置", "info": "信息", "quote": "引用"}.get(kind, kind)
        quoted = "\n".join(f"> {ln}".rstrip() for ln in body.splitlines())
        return f"> **{head}**\n>\n{quoted}\n\n"

    return ADMONITION.sub(repl, text)


def convert_figures(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        img = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", inner)
        cap = re.search(r"<figcaption>(.*?)</figcaption>", inner, re.DOTALL)
        if not img:
            return ""
        alt, src = img.group(1), img.group(2)
        out = f"![{alt}]({src})\n"
        if cap:
            c = cap.group(1).strip().replace("\n", " ")
            out += f"\n*{c}*\n"
        return out

    return FIGURE.sub(repl, text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()

    # 1) 建立 源文件 → 目标文件名 的映射
    mapping: dict[Path, str] = {}
    for i, rel in enumerate(LECTURE_ORDER, start=1):
        src = DOCS / f"{rel}.md"
        title = slug(src.read_text(encoding="utf-8").splitlines()[0])
        mapping[src.resolve()] = f"{i:02d}-{title}.md"

    for i, name in enumerate(LABS, start=1):
        src = DOCS / "labs" / f"{name}.md"
        if src.exists():
            title = slug(src.read_text(encoding="utf-8").splitlines()[0])
            mapping[src.resolve()] = f"实验{i}-{title}.md"

    for name in APPENDIX:
        src = DOCS / "appendix" / f"{name}.md"
        if src.exists():
            title = slug(src.read_text(encoding="utf-8").splitlines()[0])
            mapping[src.resolve()] = f"附录-{title}.md"

    index_src = (DOCS / "index.md").resolve()
    mapping[index_src] = "00-总览.md"

    # 2) 复制插图
    img_src = DOCS / "assets" / "img"
    img_dst = OUT / "images"
    img_dst.mkdir(exist_ok=True)
    n_img = 0
    for f in img_src.glob("*.svg"):
        shutil.copy2(f, img_dst / f.name)
        n_img += 1

    # 3) 逐个转换
    written = 0
    for src, dst_name in mapping.items():
        text = src.read_text(encoding="utf-8")

        text = convert_figures(text)
        text = convert_admonitions(text)
        # 只删图片的 { width=... } 与标题的 {: #anchor }，绝不能碰 LaTeX 的 {}
        text = re.sub(r"(\]\([^)\s]+\))\s*\{[^}]*\}", r"\1", text)
        text = re.sub(r"^(#+\s+.*?)\s*\{:[^}]*\}\s*$", r"\1", text, flags=re.M)

        # 图片路径 → images/
        text = re.sub(r"\]\((?:\.\./)+assets/img/([^)]+)\)", r"](images/\1)", text)

        # 章间链接 → 目标文件名
        def link_repl(m: re.Match[str]) -> str:
            target = (src.parent / m.group(1)).resolve()
            name = mapping.get(target)
            anchor = m.group(2) or ""
            return f"]({name}{anchor})" if name else m.group(0)

        text = LINK.sub(link_repl, text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

        (OUT / dst_name).write_text(text, encoding="utf-8")
        written += 1

    print(f"✓ 导出 {written} 个 Markdown，{n_img} 张插图 → {OUT}")
    for f in sorted(OUT.glob("*.md"))[:5]:
        print("  ", f.name)


if __name__ == "__main__":
    main()
