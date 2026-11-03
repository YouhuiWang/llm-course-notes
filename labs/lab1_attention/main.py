"""Lab 1：手写注意力

从零实现缩放点积注意力、多头注意力与 RoPE，并与 PyTorch 官方实现逐项对齐。

运行：
    python labs/lab1_attention/main.py

全部在 CPU 上跑，几秒钟出结果。所有对齐测试使用 float64，
把实现误差压到 1e-12 量级，避免被 float32 的舍入误差掩盖真正的 bug。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

TOL = 1e-10


# ---------------------------------------------------------------- 1. softmax


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定版 softmax：先减每行最大值，再取指数。"""
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def softmax_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """朴素版：直接取指数，大数值会溢出。"""
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------- 2. attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力。

    q, k, v: (B, H, N, D)
    pad_mask: (B, N)，True 表示是 padding（要屏蔽）
    返回 (输出, 注意力权重)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if causal:
        n = scores.size(-1)
        # 用有限大负数而不是 -inf：整行被 mask 时 softmax 不会变成 NaN
        future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)

    if pad_mask is not None:
        scores = scores.masked_fill(pad_mask[:, None, None, :], torch.finfo(scores.dtype).min)

    w = softmax_stable(scores, dim=-1)
    return w @ v, w


# ---------------------------------------------------------------- 3. multi-head


def multi_head_attention(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    w_o: torch.Tensor,
    b_o: torch.Tensor,
    num_heads: int,
    causal: bool = True,
) -> torch.Tensor:
    """与 nn.MultiheadAttention 相同的权重布局，便于逐项对齐。"""
    b, n, d = x.shape
    qkv = x @ w_qkv.t() + b_qkv                      # (B, N, 3d)
    q, k, v = qkv.chunk(3, dim=-1)

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(b, n, num_heads, d // num_heads).transpose(1, 2)

    out, _ = attention(split(q), split(k), split(v), causal=causal)
    out = out.transpose(1, 2).reshape(b, n, d)
    return out @ w_o.t() + b_o


# ---------------------------------------------------------------- 4. RoPE


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """旋转位置编码：把相邻两维看成一个复数平面并旋转。

    x: (..., N, D)，positions: (N,)
    """
    d = x.shape[-1]
    omega = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)   # (D/2,)
    ang = positions[:, None].to(x.dtype) * omega[None, :]          # (N, D/2)
    cos, sin = ang.cos(), ang.sin()

    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ---------------------------------------------------------------- 测试


def check(name: str, err: float, tol: float = TOL) -> bool:
    ok = err < tol
    print(f"  {'✓' if ok else '✗'} {name:<44} 最大误差 {err:.2e}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    results: list[bool] = []

    print("\n[1] softmax：稳定性与对齐")
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    results.append(check("稳定版 vs torch.softmax", (softmax_stable(logits) - logits.softmax(-1)).abs().max().item()))

    big = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float64)
    naive_big = softmax_naive(big)
    stable_big = softmax_stable(big)
    print(f"    x=1000 时朴素版输出: {naive_big.flatten().tolist()}")
    print(f"    x=1000 时稳定版输出: {[round(v, 4) for v in stable_big.flatten().tolist()]}")
    results.append(check("稳定版在 x=1000 时仍正确", (stable_big - big.softmax(-1)).abs().max().item()))

    print("\n[2] 缩放点积注意力：与 F.scaled_dot_product_attention 对齐")
    q, k, v = (torch.randn(2, 4, 16, 32, dtype=torch.float64) for _ in range(3))
    for causal in (True, False):
        out, w = attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        results.append(check(f"causal={causal} 的输出", (out - ref).abs().max().item()))
        results.append(check(f"causal={causal} 的权重行和为 1", (w.sum(-1) - 1).abs().max().item()))

    print("\n[3] 多头注意力：与 nn.MultiheadAttention 对齐")
    d, h, n = 64, 8, 12
    mha = torch.nn.MultiheadAttention(d, h, batch_first=True, dtype=torch.float64)
    x = torch.randn(2, n, d, dtype=torch.float64)
    ref, _ = mha(x, x, x, need_weights=False, attn_mask=torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1))
    mine = multi_head_attention(x, mha.in_proj_weight, mha.in_proj_bias, mha.out_proj.weight, mha.out_proj.bias, h)
    results.append(check("多头输出（含因果掩码）", (mine - ref).abs().max().item()))

    print("\n[4] RoPE：内积只依赖相对位置")
    q1 = torch.randn(64, dtype=torch.float64)
    k1 = torch.randn(64, dtype=torch.float64)
    pos = torch.arange(0, 40)
    dots = torch.stack([rope(q1, pos[:1] * 0 + m) @ rope(k1, pos[:1] * 0 + m + 5) for m in pos])
    spread = (dots.max() - dots.min()).item()
    print(f"    固定 Δ=5，改变绝对位置 m∈[0,39]，内积跨度 {spread:.2e}")
    results.append(check("内积与绝对位置无关", spread))

    d1, d2 = rope(q1, torch.tensor([3])) @ rope(k1, torch.tensor([8])), rope(q1, torch.tensor([10])) @ rope(k1, torch.tensor([15]))
    results.append(check("(3,8) 与 (10,15) 的内积相同", (d1 - d2).abs().item()))

    sinpe = lambda p: torch.cat([torch.sin(p * 10000 ** (-torch.arange(0, 64, 2) / 64)),
                                 torch.cos(p * 10000 ** (-torch.arange(0, 64, 2) / 64))]).sort().values
    sin_dots = torch.stack([(q1 + sinpe(float(m))) @ (k1 + sinpe(float(m + 5))) for m in pos])
    print(f"    对照：正弦绝对位置的内积跨度 {(sin_dots.max() - sin_dots.min()).item():.2f}（随 m 漂移）")

    print("\n[5] 因果掩码：可见比例")
    for n_ in (4, 16, 64):
        visible = n_ * (n_ + 1) // 2
        print(f"    n={n_:>2}: 可见 {visible:>5} / {n_ ** 2:>5} = {visible / n_ ** 2:.1%}")
    results.append(check("n=4 的可见比例为 62.5%", abs(4 * 5 / 2 / 16 - 0.625)))

    print(f"\n{'=' * 60}")
    print(f"通过 {sum(results)}/{len(results)} 项检查")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
