#!/bin/bash
# 双击启动：构建站点 → 后台起服务 → 打开浏览器
cd "$(dirname "$0")" || exit 1
./scripts/serve.sh start
echo
echo "（这个窗口可以关掉了，服务在后台继续运行）"
sleep 1
