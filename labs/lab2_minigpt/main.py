"""Lab 2：迷你 GPT

从零搭一个 2 层、4 头的字符级 GPT，在 CPU 上训练几分钟，
观察 loss 收敛与生成质量。所有组件都在前面讲次里出现过：
Pre-LN block、多头注意力、SwiGLU、交叉熵。

运行：
    python labs/lab2_minigpt/main.py
    python labs/lab2_minigpt/main.py --steps 400   # 训久一点
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------- 语料

CORPUS = """
大模型的发展经历了从统计语言模型到神经网络语言模型的转变。
注意力机制让模型可以并行处理序列，这是 Transformer 的核心创新。
缩放定律表明，损失随参数量和训练数据量呈幂律下降。
混合专家模型通过稀疏激活，在相同算力下容纳更多参数。
检索增强生成先用检索找到证据，再基于证据生成答案。
智能体由模型策略、工具、记忆和循环控制组成，可以完成多步任务。
量化用更低的位宽表示权重，把显存需求降到四分之一。
长上下文的关键是位置编码能否外推，而不只是窗口大小。
The quick brown fox jumps over the lazy dog.
Language models predict the next token given the previous context.
Scaling laws describe how loss decreases with compute and data.
Attention is all you need, and it changed everything.
Retrieval augmented generation grounds the answer in evidence.
Mixture of experts activates only a subset of parameters for each token.
Long context requires careful position encoding and memory management.
Quantization trades a little accuracy for a large memory saving.
Agents plan, call tools, and remember what they have already tried.
""" * 4


# ---------------------------------------------------------------- 模型


class Block(nn.Module):
    """Pre-LN block：注意力 + SwiGLU"""

    def __init__(self, d: int, h: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class MiniGPT(nn.Module):
    def __init__(self, vocab: int, d: int = 128, h: int = 4,
                 layers: int = 2, d_ff: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(1024, d)
        self.blocks = nn.ModuleList([Block(d, h, d_ff) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight          # 权重绑定

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(n, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------- 训练


def get_batch(data: torch.Tensor, ctx: int, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - ctx - 1, (bs,))
    x = torch.stack([data[i: i + ctx] for i in ix])
    y = torch.stack([data[i + 1: i + ctx + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, ctx: int,
                  bs: int, vocab: int, iters: int = 10) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, ctx, bs)
        losses.append(F.cross_entropy(model(x).view(-1, vocab), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model: nn.Module, stoi: dict, itos: dict, prompt: str,
           n_new: int = 80, temperature: float = 0.8) -> str:
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]])
    out = list(prompt)
    for _ in range(n_new):
        logits = model(idx[:, -256:])[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out.append(itos[nxt])
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    model.train()
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in CORPUS])

    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    model = MiniGPT(len(chars))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"语料 {len(CORPUS)} 字符｜词表 {len(chars)}｜参数 {n_params / 1e6:.2f}M")
    print(f"训练 {args.steps} 步，batch {args.bs}，ctx {args.ctx}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    history: list[tuple[int, float]] = []

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.ctx, args.bs)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            val = estimate_loss(model, val_data, args.ctx, args.bs, len(chars), iters=5)
            history.append((step, val))
            print(f"  step {step:>4}｜train {loss.item():.3f}｜val {val:.3f}"
                  f"｜{time.time() - t0:.0f}s")

    print(f"\n困惑度：{math.exp(history[-1][1]):.1f}")
    print("\n生成示例（temperature 0.8）：")
    for prompt in ("大模型", "The quick"):
        print(f"  [{prompt}] {sample(model, stoi, itos, prompt, n_new=60)}")

    # 存 loss 曲线
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [s for s, _ in history]
        vals = [v for _, v in history]
        plt.figure(figsize=(5, 2.6))
        plt.plot(steps, vals, color="#2f6f9f")
        plt.xlabel("step")
        plt.ylabel("val loss")
        plt.title("mini GPT loss")
        plt.tight_layout()
        plt.savefig("labs/lab2_minigpt/results/loss.png", dpi=150)
        print("\n曲线已存到 labs/lab2_minigpt/results/loss.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
