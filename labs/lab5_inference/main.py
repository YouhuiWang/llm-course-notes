"""Lab 5：推理加速

三组实测：
  1. KV cache：有缓存 vs 无缓存，逐 token 生成的耗时差
  2. 量化：INT8 后的显存占用与数值误差
  3. 投机解码：接受率如何决定加速比

运行：
    python labs/lab5_inference/main.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)


# ---------------------------------------------------------------- 1. KV cache


def gen_without_cache(k: torch.Tensor, v: torch.Tensor, steps: int = 80) -> float:
    """每步都重算整个前缀：O(n²)"""
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        q = k[:, :, i : i + 1]  # 用 k 当前行当查询，只为计时
        _ = F.scaled_dot_product_attention(q, k[:, :, : i + 1], v[:, :, : i + 1])
    return time.perf_counter() - t0


def gen_with_cache(k: torch.Tensor, v: torch.Tensor, steps: int = 80) -> float:
    """缓存历史 K/V，每步只算新 token：O(n)"""
    t0 = time.perf_counter()
    cache_k, cache_v = k[:, :, :1].clone(), v[:, :, :1].clone()
    for i in range(1, steps + 1):
        q = k[:, :, i : i + 1]
        cache_k = torch.cat([cache_k, k[:, :, i : i + 1]], dim=2)
        cache_v = torch.cat([cache_v, v[:, :, i : i + 1]], dim=2)
        _ = F.scaled_dot_product_attention(q, cache_k, cache_v)
    return time.perf_counter() - t0


# ---------------------------------------------------------------- 2. 量化


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐通道对称量化：返回 (反量化权重, scale, 零点)"""
    scale = w.abs().max(dim=1, keepdim=True).values / 127.0 + 1e-8
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q.float() * scale, scale, q


# ---------------------------------------------------------------- 3. 投机解码


def spec_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """(1 + αγ) / (1 + γc)"""
    return (1 + alpha * gamma) / (1 + gamma * cost_ratio)


def main() -> None:
    print("=" * 58)
    print("[1] KV cache：逐 token 生成 80 步")
    B, H, N, D = 1, 4, 100, 32
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    t_no = gen_without_cache(k, v)
    t_yes = gen_with_cache(k, v)
    print(f"  无缓存 {t_no * 1000:>7.1f} ms")
    print(f"  有缓存 {t_yes * 1000:>7.1f} ms")
    print(f"  加速比 {t_no / t_yes:>7.2f}×")

    print("\n" + "=" * 58)
    print("[2] 量化：2048×2048 权重矩阵")
    w = torch.randn(2048, 2048) * 0.02
    wq, scale, q = quantize_int8(w)
    err = (w - wq).abs()
    size_fp32 = w.numel() * 4 / 1e6
    size_int8 = q.numel() * 1 / 1e6
    print(
        f"  fp32 {size_fp32:>6.1f} MB  ->  int8 {size_int8:>6.1f} MB（{size_fp32 / size_int8:.0f}×）"
    )
    print(f"  平均绝对误差 {err.mean():.6f}｜最大误差 {err.max():.6f}")
    print(f"  相对误差 {(err.mean() / w.abs().mean()):.2%}")

    print("\n" + "=" * 58)
    print("[3] 投机解码：加速比随接受率变化（γ=4, 草稿代价 0.05）")
    print(f"{'接受率 α':>10}{'加速比':>10}")
    for alpha in (0.3, 0.5, 0.7, 0.9):
        print(f"{alpha:>10.1f}{spec_speedup(alpha, 4, 0.05):>10.2f}×")

    print("\n  同一接受率下，草稿越贵越不划算：")
    print(f"{'草稿代价 c':>12}{'α=0.7 的加速比':>16}")
    for c in (0.02, 0.05, 0.2, 0.5):
        print(f"{c:>12.2f}{spec_speedup(0.7, 4, c):>16.2f}×")


if __name__ == "__main__":
    main()
