# Lab 1　手写注意力

!!! abstract "实验信息"
    **对应讲次**：[第 4 讲 注意力机制完全拆解](../02-transformer/l04-attention.md)
    **耗时**：30–45 分钟　**硬件**：CPU 即可　**依赖**：`torch`

## 目标

不看公式直接写代码，把四个部件实现一遍，每实现一个就跑一次对齐测试：

| # | 部件 | 对齐对象 | 通过标准 |
|---|---|---|---|
| 1 | 数值稳定 softmax | `torch.softmax` | 误差 $<10^{-10}$，$x=1000$ 不溢出 |
| 2 | 缩放点积注意力 + 因果掩码 | `F.scaled_dot_product_attention` | 误差 $<10^{-10}$ |
| 3 | 多头注意力 | `nn.MultiheadAttention` | 误差 $<10^{-10}$ |
| 4 | RoPE | 相对位置不变性 | 内积跨度 $<10^{-10}$ |

## 运行

```bash
uv pip install --python .venv/bin/python -r requirements-lab.txt
.venv/bin/python labs/lab1_attention/main.py
```

预期输出见 `labs/lab1_attention/expected.md`，基线结果见 `labs/lab1_attention/results/report.txt`。

## 四个关键点

### 1 为什么要减最大值

输入 `[1000, 999, 998]` 时，`exp(1000)` 直接变成 `inf`，`inf/inf` 得到 `nan`。减最大值后最大的指数项变成 `exp(0)=1`——数学上等价，数值上安全。

### 2 为什么用 `finfo.min` 而不是 `-inf`

如果某一行**所有**位置都被 mask（整句都是 padding），`-inf` 会让 softmax 变成 $0/0$，输出 NaN。NaN 会顺着梯度污染整个 batch，而报错往往出现在几百步之后。

### 3 多头的 reshape 顺序

正确顺序是 `(B, N, H, D) → (B, H, N, D)`。写错了不会报错，只会静默地算错——这是最坑的一类 bug。

### 4 RoPE 为什么只依赖相对位置

$$\boldsymbol{R}_m^\top\boldsymbol{R}_n=\boldsymbol{R}_{n-m}
\quad\Longrightarrow\quad
\langle \boldsymbol{R}_m\boldsymbol{q},\boldsymbol{R}_n\boldsymbol{k}\rangle
=\boldsymbol{q}^\top\boldsymbol{R}_{n-m}\boldsymbol{k}$$

固定 $\Delta=5$、让 $m$ 从 0 扫到 39，内积应当**完全不变**；作为对照，正弦绝对位置编码的内积跨度约 7.6。

## 实测结果

```
[1] softmax  ✓ 稳定版 vs torch.softmax        最大误差 2.78e-17
[2] 注意力   ✓ causal=True 的输出             最大误差 8.88e-16
[3] 多头     ✓ 多头输出（含因果掩码）            最大误差 1.67e-16
[4] RoPE     ✓ 内积与绝对位置无关               最大误差 2.13e-14
[5] 掩码     n=64 可见 2080/4096 = 50.8%
通过 10/10 项检查
```

## 思考题

1. 把 `attention()` 里的 `math.sqrt(d_k)` 去掉，对齐测试会失败吗？如果不会，为什么？
2. `num_heads` 从 8 改到 16（保持 $d$ 不变），参数量变了吗？计算量变了吗？
3. RoPE 的 `base` 从 10000 改到 500000，对长文本有什么影响？
4. `pad_mask` 的形状写成 `(B, 1, 1, N)` 而不是 `(B, N)`，会发生什么？

## 常见失败与排查

| 现象 | 原因 |
|---|---|
| 注意力误差 $\sim10^{-2}$ | 忘了除以 $\sqrt{d_k}$ |
| 因果掩码失败、非因果通过 | `triu` 的 `diagonal` 写错，或误用 `tril` |
| 多头误差 $\sim10^{-1}$ | reshape 顺序错，head 维与 seq 维搞混 |
| 出现 `nan` | 用了 `-inf` 做 mask，或朴素 softmax 溢出 |
| RoPE 跨度不为 0 | 角度没按位置缩放，或奇偶维配错 |

## 交付

- `main.py` 一次运行通过（10/10）
- 回答思考题 1–4
- 附一张注意力权重热图（任意输入即可）
