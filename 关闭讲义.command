#!/bin/bash
# 双击关闭：停止后台服务
cd "$(dirname "$0")" || exit 1
./scripts/serve.sh stop
echo
sleep 1
