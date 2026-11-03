#!/usr/bin/env python3
"""把「全书打印版」打包成一个自包含的单文件 HTML。

双击即可打开，不需要启动任何服务：
  - CSS 内联（含 KaTeX 字体，转成 data URI）
  - JS 内联（去掉 Material 的运行时，避免 file:// 下的 fetch 报错）
  - 插图内联（SVG 转 data URI）
  - 侧栏/顶栏隐藏，只留正文

输出：dist/大模型发展与相关技术详解-单页版.html
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SRC = SITE / "print_page" / "index.html"
OUT = ROOT / "dist" / "大模型发展与相关技术详解-单页版.html"

# 这些脚本在 file:// 下会尝试 fetch 本地文件（搜索索引等），直接丢掉
DROP_SCRIPTS = ("bundle.", "lunr", "workers")

EXTRA_CSS = """
/* 单页版：只留正文 */
.md-header, .md-sidebar, .md-footer, .md-top, .md-search { display: none !important; }
.md-main__inner { max-width: 52rem; margin: 0 auto; }
.md-content { max-width: 100%; }
body { background: #fff; }
@media print { .md-typeset h1 { break-before: page; } }
"""


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def inline_css_urls(css: str, base: Path) -> str:
    """把 CSS 里的 url(...) 换成 data URI（字体、图片）"""

    def repl(m: re.Match[str]) -> str:
        raw = m.group(1).strip("'\"")
        if raw.startswith(("data:", "http://", "https://", "#")):
            return m.group(0)
        f = (base / raw).resolve()
        if not f.exists():
            return m.group(0)
        mime = {
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }.get(f.suffix.lower(), "application/octet-stream")
        return f"url({data_uri(f, mime)})"

    return re.sub(r"url\(([^)]+)\)", repl, css)


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    base = SRC.parent

    # 1) 内联样式表
    def css_repl(m: re.Match[str]) -> str:
        href = m.group(1)
        f = (base / href).resolve()
        if not f.exists():
            return ""
        css = inline_css_urls(f.read_text(encoding="utf-8"), f.parent)
        return f"<style>\n{css}\n</style>"

    html = re.sub(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"[^>]*>', css_repl, html)

    # 2) 内联脚本（跳过会 fetch 的）
    def js_repl(m: re.Match[str]) -> str:
        src = m.group(1)
        if any(k in src for k in DROP_SCRIPTS):
            return ""
        f = (base / src).resolve()
        if not f.exists():
            return ""
        return f"<script>\n{f.read_text(encoding='utf-8')}\n</script>"

    html = re.sub(r'<script[^>]+src="([^"]+)"[^>]*>\s*</script>', js_repl, html)

    # 3) 插图转 data URI
    def img_repl(m: re.Match[str]) -> str:
        src = m.group(2)
        f = (base / src).resolve()
        if not f.exists():
            return m.group(0)
        mime = "image/svg+xml" if f.suffix == ".svg" else "image/png"
        return m.group(0).replace(src, data_uri(f, mime))

    html = re.sub(r'<img([^>]+)src="([^"]+)"', img_repl, html)

    # 4) 追加单页样式
    html = html.replace("</head>", f"<style>{EXTRA_CSS}</style>\n</head>", 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    size = OUT.stat().st_size / 1e6
    print(f"✓ {OUT.relative_to(ROOT)}（{size:.1f} MB）")
    print("  双击即可打开，无需启动服务")


if __name__ == "__main__":
    main()
