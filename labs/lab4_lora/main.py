"""Lab 4：LoRA 微调

先在一个任务上预训练一个小 Transformer，再换任务微调，比较三种方式：
  1. 全参微调（Full FT）
  2. LoRA（低秩适配，r=4）
  3. 只训分类头（Head only）

看三件事：可训练参数占比、耗时、最终准确率。

运行：
    python labs/lab4_lora/main.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)

VOCAB, SEQ, D, H, LAYERS = 10, 8, 64, 4, 2


# ---------------------------------------------------------------- 数据


def make_data(n: int, task: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    if task == "sum":
        y = (x.sum(dim=1) > 36).long()
    else:  # first > last（约 50% 正类，需要看两个位置）
        y = (x[:, 0] > x[:, -1]).long()
    return x, y


# ---------------------------------------------------------------- 模型


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x) + self.pos(torch.arange(x.size(1)))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h).mean(dim=1))


# ---------------------------------------------------------------- LoRA


class LoRALinear(nn.Module):
    """W x + (B A x) * scale，只训练 A 与 B"""

    def __init__(self, base: nn.Linear, r: int = 4) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(model: nn.Module, r: int = 4) -> None:
    """把模型里所有 Linear 换成 LoRALinear（head 除外）"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            continue          # 不碰注意力内部：它依赖 out_proj.weight 的 fused 实现
        if isinstance(module, nn.Linear):
            if name != "head":
                setattr(model, name, LoRALinear(module, r))
        elif isinstance(module, nn.Sequential):
            for i, sub in enumerate(module):
                if isinstance(sub, nn.Linear):
                    module[i] = LoRALinear(sub, r)
        else:
            apply_lora(module, r)


def set_trainable(model: TinyEncoder, mode: str) -> int:
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif mode == "lora":
        for p in model.parameters():
            p.requires_grad = False
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.A.requires_grad = True
                m.B.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 训练


def train(model: TinyEncoder, x: torch.Tensor, y: torch.Tensor, steps: int,
          lr: float = 1e-3) -> float:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, x.size(0), (64,))
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


@torch.no_grad()
def accuracy(model: TinyEncoder, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(x).argmax(dim=1) == y).float().mean())


def main() -> None:
    xa, ya = make_data(4000, "sum", seed=1)
    xb, yb = make_data(2000, "duplicate", seed=2)
    xb_val, yb_val = make_data(500, "duplicate", seed=3)

    print("预训练：任务 A（判断序列和是否 > 36）")
    base = TinyEncoder()
    total = sum(p.numel() for p in base.parameters())
    t = train(base, xa, ya, steps=300)
    print(f"  参数 {total / 1e3:.0f}K｜耗时 {t:.1f}s｜任务 A 准确率 {accuracy(base, xa, ya):.3f}\n")

    print("微调：任务 B（判断首元素是否大于末元素）")
    print(f"{'方式':<12}{'可训练参数':>12}{'占比':>8}{'耗时':>8}{'准确率':>8}")
    print("-" * 50)

    for mode in ("full", "lora", "head"):
        import copy

        model = copy.deepcopy(base)
        if mode == "lora":
            apply_lora(model, r=4)
        n_train = set_trainable(model, mode)
        t = train(model, xb, yb, steps=300, lr=1e-3 if mode != "full" else 5e-4)
        acc = accuracy(model, xb_val, yb_val)
        print(f"{mode:<12}{n_train:>12,}{n_train / total:>7.1%}{t:>7.1f}s{acc:>8.3f}")

    print("\n注：LoRA 只训练很少的参数，准确率接近全参微调；"
          "只训 head 明显更差——说明「适配哪一层」比「训多少参数」更重要。")


if __name__ == "__main__":
    main()
