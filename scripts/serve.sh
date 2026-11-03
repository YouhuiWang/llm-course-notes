#!/usr/bin/env bash
# 本地预览服务的启动 / 停止 / 状态控制
#
#   ./scripts/serve.sh start    构建站点并在后台启动服务，然后打开浏览器
#   ./scripts/serve.sh stop     停止服务
#   ./scripts/serve.sh status   查看状态
#
# 端口默认 8123，可用 PORT=9000 ./scripts/serve.sh start 覆盖。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8123}"
URL="http://127.0.0.1:${PORT}"
CACHE="$ROOT/.cache"
PIDFILE="$CACHE/serve.pid"
LOGFILE="$CACHE/serve.log"

mkdir -p "$CACHE"

running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_busy() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
}

build() {
  if [[ -x "$ROOT/.venv/bin/mkdocs" ]]; then
    echo "▸ 重新构建站点 …"
    ( cd "$ROOT" && "$ROOT/.venv/bin/mkdocs" build --strict >/dev/null )
  elif [[ -d "$ROOT/site" ]]; then
    echo "▸ 未找到虚拟环境，使用已有的 site/"
  else
    echo "✗ 既没有 .venv 也没有 site/，请先运行："
    echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"
    exit 1
  fi
}

start() {
  if running; then
    echo "✓ 服务已在运行：$URL"
    open "$URL" >/dev/null 2>&1 &
    disown 2>/dev/null || true
    exit 0
  fi
  if port_busy; then
    echo "✗ 端口 ${PORT} 已被其它程序占用"
    echo "  换一个端口：PORT=9000 ./scripts/serve.sh start"
    exit 1
  fi

  build
  rm -f "$PIDFILE"

  # 完全脱离当前会话：三个 fd 都重定向，避免父进程退出时被一起回收
  (
    cd "$ROOT/site" || exit 1
    nohup python3 -m http.server "$PORT" --bind 127.0.0.1 \
      </dev/null >"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
    disown 2>/dev/null || true
  )

  for _ in $(seq 1 40); do
    port_busy && break
    sleep 0.2
  done

  if ! port_busy; then
    echo "✗ 启动失败，日志：$LOGFILE"
    tail -5 "$LOGFILE" 2>/dev/null || true
    exit 1
  fi

  echo "✓ 已启动：$URL"
  echo "  关闭：双击「关闭讲义.command」，或运行 ./scripts/serve.sh stop"
  # 后台打开浏览器，避免在无图形会话里阻塞
  open "$URL" >/dev/null 2>&1 &
  disown 2>/dev/null || true
}

stop() {
  if running; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "✓ 已停止"
  else
    rm -f "$PIDFILE"
    if port_busy; then
      echo "⚠ 端口 ${PORT} 仍被其它程序占用（不是本脚本启动的）"
    else
      echo "· 服务本来就没在运行"
    fi
  fi
}

status() {
  if running; then
    echo "运行中：${URL}（PID $(cat "$PIDFILE")）"
  else
    echo "未运行"
  fi
}

case "${1:-}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  *) echo "用法：$0 {start|stop|status}"; exit 1 ;;
esac
