# Lab 1　手写注意力

> 配套：[第 4 讲　注意力机制完全拆解](../../docs/02-transformer/l04-attention.md)
> 目标：**不看公式直接写代码**，把缩放、掩码、多头、RoPE 全部实现一遍，并与 PyTorch 逐项对齐。

## 你要做什么

从零实现四个部件，每实现一个就跑一次对齐测试：

| # | 部件 | 对齐对象 | 通过标准 |
|---|---|---|---|
| 1 | 数值稳定 softmax | `torch.softmax` | 误差 < $10^{-10}$，且 $x=1000$ 时不溢出 |
| 2 | 缩放点积注意力 + 因果掩码 | `F.scaled_dot_product_attention` | 误差 < $10^{-10}$ |
| 3 | 多头注意力 | `nn.MultiheadAttention` | 误差 < $10^{-10}$ |
| 4 | RoPE | 数学性质（相对位置不变） | 内积跨度 < $10^{-10}$ |

## 怎么跑

```bash
cd 大模型发展与相关技术详解
uv pip install --python .venv/bin/python -r requirements-lab.txt
.venv/bin/python labs/lab1_attention/main.py
```

预期输出见 [`expected.md`](expected.md)，完整基线见 [`results/report.txt`](results/report.txt)。

## 四个关键点

### 1 为什么要减最大值

```python
def softmax_naive(x):
    e = x.exp()
    return e / e.sum(-1, keepdim=True)
```

输入是 `[1000, 999, 998]` 时，`exp(1000)` 直接变成 `inf`，`inf/inf` 得到 `nan`。

减去最大值后，最大的指数项变成 `exp(0)=1`，其余都是小数——数学上等价，数值上安全。

### 2 为什么用 `finfo.min` 而不是 `-inf`

```python
scores.masked_fill(future, torch.finfo(scores.dtype).min)   # ✓
scores.masked_fill(future, float("-inf"))                   # ✗
```

如果某一行的**所有**位置都被 mask（比如整句都是 padding），`-inf` 会让 softmax 变成 $0/0$，输出 NaN。NaN 会顺着梯度污染整个 batch，而报错往往出现在几百步之后。

### 3 多头的 reshape 顺序

正确顺序：`(B, N, H, D) → (B, H, N, D)`。

```python
t.view(b, n, h, d // h).transpose(1, 2)   # ✓
t.view(b, h, n, d // h)                    # ✗ 会把 head 维和序列维搞混
```

写错了不会报错，只会静默地算错——这是最坑的一类 bug。

### 4 RoPE 为什么只依赖相对位置

$$\boldsymbol{R}_m^\top\boldsymbol{R}_n=\boldsymbol{R}_{n-m}$$

所以 $\langle \boldsymbol{R}_m\boldsymbol{q},\ \boldsymbol{R}_n\boldsymbol{k}\rangle=\boldsymbol{q}^\top\boldsymbol{R}_{n-m}\boldsymbol{k}$，内积只和 $n-m$ 有关。

实验里固定 $\Delta=5$，让绝对位置 $m$ 从 0 扫到 39，内积应该**完全不变**；作为对照，正弦绝对位置编码的内积会随 $m$ 漂移（跨度约 7.6）。

## 思考题

1. 把 `attention()` 里的 `math.sqrt(d_k)` 去掉，对齐测试会失败吗？如果不会，为什么？
2. 多头的 `num_heads` 从 8 改到 16（保持 $d$ 不变），参数量变了吗？计算量变了吗？
3. RoPE 的 `base` 从 10000 改到 500000，对长文本有什么影响？
4. 如果 `pad_mask` 的形状写成 `(B, 1, 1, N)` 而不是 `(B, N)`，会发生什么？

## 提交要求

- `main.py` 一次运行通过（10/10）
- 回答思考题 1–4（每题 3–5 句）
- 附一张你画的注意力权重热图（任意输入即可）
