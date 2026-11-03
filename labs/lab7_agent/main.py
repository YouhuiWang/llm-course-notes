"""Lab 7：Agent 循环与消融

在模拟环境里实现一个 ReAct agent，并做五组对比：
  1. baseline（记忆 + 反思 + 并行工具）
  2. 去反思：出错后原样重试，容易卡死
  3. 去记忆：只看最近两步，会重复读已读文件
  4. 去并行：一次只能读一个文件
  5. 小模型：错误率更高

不调用任何 API：环境是内存里的假仓库，"模型"是一个带噪声的规则策略。
这样每次运行都可复现，消融的差异才看得清。

运行：
    python labs/lab7_agent/main.py
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 环境

REPO = {
    "src/main.py": 120,
    "src/model.py": 340,
    "src/data.py": 85,
    "src/utils.py": 210,
    "tests/test_model.py": 160,
    "tests/test_data.py": 95,
    "scripts/train.py": 430,
    "docs/conf.py": 40,
}
TODO_FILES = {"src/model.py", "scripts/train.py", "src/utils.py"}

TASKS = [
    ("统计所有 .py 文件的总行数", "lines"),
    ("找出最大的 .py 文件", "largest"),
    ("统计带 TODO 注释的文件数", "todo"),
]


@dataclass
class ToolResult:
    ok: bool
    output: str


class Sandbox:
    """假仓库：三个工具，纯函数，调用次数可统计"""

    def __init__(self) -> None:
        self.calls = 0

    def search_files(self, pattern: str) -> ToolResult:
        self.calls += 1
        if not pattern.endswith(".py"):
            return ToolResult(False, "PATTERN_INVALID")
        return ToolResult(True, "、".join(sorted(REPO)))

    def read_file(self, path: str) -> ToolResult:
        self.calls += 1
        if path not in REPO:
            return ToolResult(False, f"NOT_FOUND {path}")
        return ToolResult(True, f"{path}:{REPO[path]}")

    def run_python(self, code: str) -> ToolResult:
        self.calls += 1
        if "TODO" in code:
            return ToolResult(True, f"TODO {len(TODO_FILES)}")
        return ToolResult(False, "SYNTAX_ERROR")


# ---------------------------------------------------------------- 策略


@dataclass
class Action:
    tool: str
    arg: str

    def key(self) -> str:
        return f"{self.tool}:{self.arg}"


@dataclass
class Config:
    name: str
    use_memory: bool = True
    use_reflection: bool = True
    parallel: int = 3
    error_rate: float = 0.15


class Policy:
    """带噪声的规则策略：模拟一个会犯错的 LLM"""

    def __init__(self, cfg: Config, seed: int) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.last_failed: Action | None = None

    def act(self, kind: str, read: set[str], searched: bool, step: int = 0) -> list[Action]:
        # 不反思：上次失败就原样重试（容易死循环）
        if self.last_failed is not None and not self.cfg.use_reflection:
            return [self.last_failed]

        if not searched:
            return [Action("search", "*.py")]

        remaining = [p for p in sorted(REPO) if p not in read]

        # 按错误率挑一个无效动作
        if self.rng.random() < self.cfg.error_rate:
            pool = [Action("read", "src/missing.py"), Action("run_python", "print(1)")]
            if read:
                pool.append(Action("read", sorted(read)[0]))  # 重复读
            return [self.rng.choice(pool)]

        if remaining:
            # 没有记忆时，读了几步就"猜"自己读完了——这是它失败的根源
            if not self.cfg.use_memory and step >= 8:
                return [Action("finish", "guess")]
            return [Action("read", p) for p in remaining[: self.cfg.parallel]]

        if kind == "todo":
            return [Action("run_python", "count TODO")]
        return [Action("finish", "done")]


# ---------------------------------------------------------------- 循环


def run_agent(kind: str, cfg: Config, seed: int, max_steps: int = 20) -> dict:
    sandbox = Sandbox()
    policy = Policy(cfg, seed)
    log: list[tuple[Action, bool]] = []
    tokens = 0
    stuck = 0
    last_read = 0

    for step in range(1, max_steps + 1):
        # 可见的历史：有记忆看全部，没记忆只看最近两步
        visible = log if cfg.use_memory else log[-6:]
        read = {a.arg for a, ok in visible if a.tool == "read" and ok}
        searched = any(a.tool == "search" and ok for a, ok in visible)

        actions = policy.act(kind, read, searched, step)
        tokens += 200 + 45 * len(log)  # 粗略模拟上下文增长

        for act in actions:
            if act.tool == "search":
                r = sandbox.search_files(act.arg)
            elif act.tool == "read":
                r = sandbox.read_file(act.arg)
            elif act.tool == "run_python":
                r = sandbox.run_python(act.arg)
            else:  # finish
                all_read = len({a.arg for a, ok in log if a.tool == "read" and ok}) == len(REPO)
                r = ToolResult(all_read, "done" if all_read else "NOT_ENOUGH_INFO")

            log.append((act, r.ok))
            policy.last_failed = None if r.ok else act

            if r.ok and act.tool in ("run_python", "finish"):
                return {"ok": True, "steps": step, "tokens": tokens, "calls": sandbox.calls}

        # 无进展检测：连续 4 步没有新增已读文件
        n_read = len({a.arg for a, ok in log if a.tool == "read" and ok})
        stuck = stuck + 1 if n_read == last_read else 0
        last_read = n_read
        if stuck >= 4:
            return {
                "ok": False,
                "steps": step,
                "tokens": tokens,
                "calls": sandbox.calls,
                "reason": "no_progress",
            }

    return {
        "ok": False,
        "steps": max_steps,
        "tokens": tokens,
        "calls": sandbox.calls,
        "reason": "max_steps",
    }


# ---------------------------------------------------------------- 主流程


CONFIGS = [
    Config("baseline"),
    Config("去反思", use_reflection=False),
    Config("去记忆", use_memory=False),
    Config("去并行", parallel=1),
    Config("小模型（错误率 0.35）", error_rate=0.35),
]

RUNS = 24


def main() -> None:
    print(f"环境：{len(REPO)} 个文件、{len(TASKS)} 个任务，每个配置跑 {RUNS} 次\n")
    header = f"{'配置':<22}{'成功率':>8}{'平均步数':>10}{'平均 token':>12}{'工具调用':>10}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        ok = steps = toks = calls = 0
        for i in range(RUNS):
            _, kind = TASKS[i % len(TASKS)]
            r = run_agent(kind, cfg, seed=i)
            ok += r["ok"]
            steps += r["steps"]
            toks += r["tokens"]
            calls += r["calls"]
        print(
            f"{cfg.name:<22}{ok / RUNS:>8.0%}{steps / RUNS:>10.1f}"
            f"{toks / RUNS:>12.0f}{calls / RUNS:>10.1f}"
        )

    print("\n注：去并行/小模型的步数与 token 明显更高；去反思在遇到错误时容易卡住。")


if __name__ == "__main__":
    main()
