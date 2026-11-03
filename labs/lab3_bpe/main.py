"""Lab 3：手写 BPE

从零实现字节对编码：统计相邻符号对的频次，反复合并最高频的一对，
直到词表达到目标大小。然后验证编码-解码无损，并对比压缩率。

运行：
    python labs/lab3_bpe/main.py
"""

from __future__ import annotations

import re
from collections import Counter

CORPUS = """
大模型的发展经历了从统计语言模型到神经网络语言模型的转变。
注意力机制让模型可以并行处理序列，这是 Transformer 的核心创新。
缩放定律表明，损失随参数量和训练数据量呈幂律下降。
混合专家模型通过稀疏激活，在相同算力下容纳更多参数。
检索增强生成先用检索找到证据，再基于证据生成答案。
智能体由模型策略、工具、记忆和循环控制组成，可以完成多步任务。
The quick brown fox jumps over the lazy dog.
Language models predict the next token given the previous context.
Scaling laws describe how loss decreases with compute and data.
Attention is all you need, and it changed everything.
Retrieval augmented generation grounds the answer in evidence.
Retrieval augmented generation grounds the answer in evidence.
Mixture of experts activates only a subset of parameters for each token.
Long context requires careful position encoding and memory management.
Quantization trades a little accuracy for a large memory saving.
Agents plan, call tools, and remember what they have already tried.
大模型的训练需要海量数据与算力，推理则受显存带宽限制。
长上下文的关键是位置编码能否外推，而不只是窗口大小。
量化用更低的位宽表示权重，把显存需求降低到四分之一。
智能体在长任务上容易累积误差，所以要缩短链路并加检查点。
评测不仅要看准确率，还要看数据污染与裁判偏差。
"""


def pretokenize(text: str) -> list[str]:
    """按空白与标点切分，保留中文单字"""
    return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z]+|[0-9]+|[^\s]", text)


def train_bpe(text: str, vocab_size: int = 300) -> tuple[list[tuple[str, str]], list[str]]:
    """返回 (合并规则列表, 初始符号表)"""
    words = pretokenize(text)
    freq = Counter(words)
    # 每个词表示为字符元组
    corpus = {tuple(w): c for w, c in freq.items()}

    symbols = sorted({ch for w in corpus for ch in w})
    merges: list[tuple[str, str]] = []

    while len(symbols) < vocab_size:
        pairs: Counter[tuple[str, str]] = Counter()
        for word, c in corpus.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (best, _) = pairs.most_common(1)[0]
        merges.append(best)

        merged = best[0] + best[1]
        symbols.append(merged)
        corpus = {_merge(word, best): c for word, c in corpus.items()}

    return merges, symbols


def _merge(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """按合并顺序贪心编码"""
    rank = {pair: i for i, pair in enumerate(merges)}
    tokens: list[str] = []
    for w in pretokenize(text):
        sym = list(w)
        while len(sym) > 1:
            pairs = [(sym[i], sym[i + 1]) for i in range(len(sym) - 1)]
            # 选排名最靠前（最早被合并）的那一对
            best = min(pairs, key=lambda p: rank.get(p, 10**9))
            if best not in rank:
                break
            sym = list(_merge(tuple(sym), best))
        tokens.extend(sym)
    return tokens


def main() -> None:
    print(f"语料：{len(CORPUS)} 字符，{len(pretokenize(CORPUS))} 个预分词单元\n")

    for vocab in (250, 350, 450):
        merges, symbols = train_bpe(CORPUS, vocab)
        toks = encode(CORPUS, merges)
        joined = "".join(toks)

        lossless = joined == "".join(pretokenize(CORPUS))
        chars = len("".join(pretokenize(CORPUS)))
        print(
            f"词表 {len(symbols):>4}｜合并 {len(merges):>4} 次｜"
            f"token 数 {len(toks):>5}｜每 token {chars / len(toks):.2f} 字符｜"
            f"无损 {'✓' if lossless else '✗'}"
        )

    merges, _ = train_bpe(CORPUS, 300)
    print("\n前 10 条合并规则：")
    for i, (a, b) in enumerate(merges[:10], 1):
        print(f"  {i:>2}. {a!r} + {b!r} -> {(a + b)!r}")

    sample = "大模型的发展经历了从统计语言模型"
    print(f"\n编码示例：{sample}")
    print(f"  -> {encode(sample, merges)}")

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        n = len(enc.encode(CORPUS))
        print(f"\n对照 tiktoken（cl100k）：{n} token，本实现 {len(encode(CORPUS, merges))} token")
    except Exception:
        print("\n（未安装 tiktoken，跳过对照）")


if __name__ == "__main__":
    main()
