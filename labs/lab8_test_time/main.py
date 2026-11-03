"""Lab 8：测试时计算

三件事：
  1. 多数投票：单次正确率 p、采样 k 次，模拟与理论对照
  2. 过度思考：思考 token 增加时，准确率先升后降
  3. 成本账：每答对一题的成本，随 k 怎么变

纯计算，不需要模型，一秒出结果。

运行：
    python labs/lab8_test_time/main.py
"""

from __future__ import annotations

from math import comb

import numpy as np

RNG = np.random.default_rng(0)


def majority_theory(p: float, k: int) -> float:
    """理论：至少 k//2+1 次正确的概率"""
    need = k // 2 + 1
    return sum(comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(need, k + 1))


def majority_sim(p: float, k: int, trials: int = 20000) -> float:
    """模拟：采 k 次，多数投票"""
    samples = RNG.random((trials, k)) < p
    votes = samples.sum(axis=1)
    return float((votes >= k // 2 + 1).mean())


def overthinking(tokens: int, peak: int = 4096) -> float:
    """准确率随思考长度的曲线：先升后降"""
    rise = 0.42 + 0.30 * (1 - np.exp(-tokens / 1500))
    decay = 0.11 * np.log2(max(tokens, peak) / peak) if tokens > peak else 0.0
    return float(rise - decay)


def main() -> None:
    print("=" * 62)
    print("[1] 多数投票：理论 vs 模拟")
    print(f"{'p':>5}{'k':>5}{'理论':>10}{'模拟':>10}{'提升':>10}")
    for p in (0.4, 0.6, 0.8):
        for k in (1, 5, 9, 15):
            t = majority_theory(p, k)
            s = majority_sim(p, k)
            print(f"{p:>5.1f}{k:>5}{t:>10.3f}{s:>10.3f}{t - p:>+10.3f}")
        print()

    print("=" * 62)
    print("[2] 过度思考：准确率 vs 思考 token 数")
    print(f"{'tokens':>8}{'准确率':>10}")
    for t in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        print(f"{t:>8}{overthinking(t):>10.3f}")

    print("\n" + "=" * 62)
    print("[3] 成本账：每答对一题要花多少")
    print(f"{'k':>4}{'准确率':>10}{'总成本':>10}{'每题成本':>12}")
    p, unit = 0.6, 1.0
    for k in (1, 3, 5, 9, 15):
        acc = majority_theory(p, k)
        total = k * unit
        print(f"{k:>4}{acc:>10.3f}{total:>10.1f}{total / acc:>12.2f}")

    print("\n结论：k 从 1 到 5，准确率 0.60→0.68，每题成本 1.67→7.33；"
          "再往上收益迅速递减。")


if __name__ == "__main__":
    main()
