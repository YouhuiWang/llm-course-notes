#!/usr/bin/env bash
# 用 Chrome headless 把"全书打印版"页面打印成 PDF。
# 优点：admonition / Mermaid / KaTeX 全部保留，所见即所得。
# 依赖：Google Chrome（macOS 默认路径），或设置 CHROME 环境变量。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/site/print_page/index.html"
OUT_DIR="$ROOT/dist"
OUT="$OUT_DIR/大模型发展与相关技术详解.pdf"

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [[ ! -x "$CHROME" ]]; then
  echo "✗ 未找到 Chrome：$CHROME" >&2
  echo "  请安装 Chrome，或设置环境变量 CHROME=/path/to/chrome" >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "✗ 找不到打印版页面：$SRC（先运行 make build）" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "▸ 打印 PDF …"
"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --virtual-time-budget=30000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$OUT" \
  "file://$SRC" 2>/dev/null

if [[ -f "$OUT" ]]; then
  SIZE=$(du -h "$OUT" | cut -f1)
  PAGES=$(strings "$OUT" | grep -c "/Type /Page" || true)
  echo "✓ ${OUT}（${SIZE}，约 ${PAGES} 页）"
else
  echo "✗ PDF 生成失败" >&2
  exit 1
fi
