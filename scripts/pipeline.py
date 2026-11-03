#!/usr/bin/env python3
"""单讲交付流水线：按顺序执行，每步校验通过才继续。

用法：
    python scripts/pipeline.py l16              # 跑 L16 的全流程
    python scripts/pipeline.py l16 --commit     # 通过后自动提交
    python scripts/pipeline.py --all            # 对所有已撰写的讲跑一遍
    python scripts/pipeline.py l16 --skip-pdf   # 跳过 PDF（快速迭代时用）

七个步骤，任一失败立即退出（退出码 1）：
    1. 内容检查   文件存在、非占位、图/算例/代码齐全
    2. 出图       运行 scripts/figs/l{NN}_*.py，逐个校验产物
    3. 审计       scripts/audit.py（结构、图号、图注、公式、死链）
    4. 构建       mkdocs build --strict（0 warning）
    5. 死链       scripts/check_links.py
    6. PDF        scripts/build_pdf.sh + 页数校验
    7. 提交       git commit（需 --commit）
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
MKDOCS = str(ROOT / ".venv" / "bin" / "mkdocs")
DOCS = ROOT / "docs"
FIGS = ROOT / "scripts" / "figs"

TOTAL = 7


class StepFailed(Exception):
    """某一步校验失败"""


def banner(i: int, name: str) -> None:
    print(f"\n▸ [{i}/{TOTAL}] {name}")


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(cmd: list[str], cwd: Path = ROOT, tail: int = 12) -> str:
    """执行命令，失败时打印尾部输出并抛错。"""
    env = {**os.environ, "NO_MKDOCS_2_WARNING": "1"}
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print("\n".join(out.strip().splitlines()[-tail:]))
        raise StepFailed(f"命令失败（退出码 {proc.returncode}）：{' '.join(cmd[:3])} …")
    return out


def find_lecture(key: str) -> Path:
    matches = [p for p in DOCS.rglob("l[0-9][0-9]-*.md") if key in p.stem]
    if len(matches) != 1:
        raise StepFailed(f"找不到唯一匹配的讲次：{key}（命中 {len(matches)} 个）")
    return matches[0]


# ------------------------------------------------------------------ 步骤


def step_content(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "本讲正在撰写中" in text:
        raise StepFailed("仍是占位页，先撰写正文")

    body = re.sub(r"```.*?```", "", text, flags=re.S)
    chars = len(re.findall(r"[\u4e00-\u9fa5]", body))
    figs = len(re.findall(r"<figcaption>", text))
    examples = len(re.findall(r"^### 例 ", text, flags=re.M))
    codes = len(re.findall(r"^```python", text, flags=re.M))

    print(f"    字数 {chars}｜图 {figs}｜算例 {examples}｜代码块 {codes}")
    if chars < 1200:
        raise StepFailed(f"正文过短（{chars} 字），至少 1200")
    if figs < 4:
        raise StepFailed(f"图不足（{figs} 张），至少 4")
    if examples < 2:
        raise StepFailed(f"算例不足（{examples} 个），至少 2")
    if codes < 1:
        raise StepFailed("缺少可运行代码块")


def step_figures(key: str) -> None:
    scripts = sorted(FIGS.glob(f"{key}_*.py"))
    if not scripts:
        raise StepFailed(f"没有找到 {key}_*.py 的绘图脚本")
    for s in scripts:
        run([PY, str(s)], cwd=FIGS)
        svg = DOCS / "assets" / "img" / (s.stem.replace("_", "-") + ".svg")
        if not svg.exists():
            raise StepFailed(f"脚本跑完但没生成 {svg.name}")
    print(f"    共 {len(scripts)} 张图，全部生成")


def step_audit(stem: str) -> None:
    run([PY, "scripts/audit.py", stem])
    print("    结构审计通过")


def step_build() -> None:
    out = ANSI.sub("", run([MKDOCS, "build", "--strict"]))
    # 只认 MkDocs 自己的告警行，避免误伤横幅文案里的 "NO_MKDOCS_2_WARNING"
    warnings = len(re.findall(r"^\s*WARNING\s+-", out, flags=re.M))
    if warnings:
        raise StepFailed(f"构建有 {warnings} 条 warning")
    print("    --strict 构建通过")


def step_links() -> None:
    run([PY, "scripts/check_links.py"])
    print("    死链检查通过")


def step_pdf() -> None:
    run(["bash", "scripts/build_pdf.sh"])
    pdf = ROOT / "dist" / "大模型发展与相关技术详解.pdf"
    if not pdf.exists() or pdf.stat().st_size < 500_000:
        raise StepFailed("PDF 未生成或体积异常")
    print(f"    PDF {pdf.stat().st_size / 1e6:.1f} MB")


def step_commit(key: str, enabled: bool) -> None:
    if not enabled:
        print("    跳过（加 --commit 启用）")
        return
    run(["git", "add", "-A"])
    run(["git", "-c", "user.name=wyh", "-c", "user.email=wyh@local",
         "commit", "-q", "-m", f"pipeline: {key} 交付"])
    out = run(["git", "log", "--oneline", "-1"])
    print(f"    {out.strip()}")


# ------------------------------------------------------------------ 主流程


def pipeline(key: str, commit: bool, skip_pdf: bool) -> None:
    path = find_lecture(key)
    stem = path.stem
    print(f"════ 流水线：{stem} ════")

    steps = [
        ("内容检查", lambda: step_content(path)),
        ("出图", lambda: step_figures(stem.split("-")[0])),
        ("结构审计", lambda: step_audit(stem)),
        ("构建", step_build),
        ("死链", step_links),
    ]
    if not skip_pdf:
        steps.append(("PDF", step_pdf))
    steps.append(("提交", lambda: step_commit(stem, commit)))

    for i, (name, fn) in enumerate(steps, start=1):
        banner(i, name)
        fn()
        print(f"    ✓ {name} 通过")

    print(f"\n════ {stem} 全部通过 ════")


def main() -> int:
    ap = argparse.ArgumentParser(description="单讲交付流水线")
    ap.add_argument("lecture", nargs="?", help="讲次关键字，如 l16")
    ap.add_argument("--all", action="store_true", help="对所有已撰写的讲依次执行")
    ap.add_argument("--commit", action="store_true", help="通过后自动 git commit")
    ap.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 步骤")
    args = ap.parse_args()

    try:
        if args.all:
            keys = []
            for p in sorted(DOCS.rglob("l[0-9][0-9]-*.md")):
                if "本讲正在撰写中" not in p.read_text(encoding="utf-8"):
                    keys.append(p.stem)
            print(f"共 {len(keys)} 讲待检：{', '.join(keys)}\n")
            for k in keys:
                pipeline(k, args.commit, args.skip_pdf)
        else:
            if not args.lecture:
                ap.error("需要指定讲次，或用 --all")
            pipeline(args.lecture, args.commit, args.skip_pdf)
    except StepFailed as e:
        print(f"\n✗ 中止：{e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
