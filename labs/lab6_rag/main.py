"""Lab 6：最小 RAG

把 RAG 的每一环摊开实现：切分 → 混合检索（BM25 + TF-IDF/SVD）→ 重排 →
抽取式生成 → 四指标评估，并做消融对比。

不依赖任何预训练模型或外部 API：
  - 稀疏检索：自己实现的 BM25
  - 稠密检索：TF-IDF + SVD（LSA），等价于一个"没训练过的嵌入模型"
  - 生成：抽取式模板（挑最相关的一句 + 一条"模型自己补的"内容）
  - 评估：规则化的 RAGAS 近似（词面覆盖）

运行：
    python labs/lab6_rag/main.py
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- 语料

DOCS = [
    (
        "attention",
        "注意力机制",
        """
注意力用查询与键的相似度对值加权求和。点积的方差随维度线性增长，
所以必须除以维度的平方根，否则 softmax 会饱和到 one-hot，梯度消失。
多头注意力把表示空间切成若干子空间，每个头学一种关系模式。
因果掩码把上三角置为负无穷，保证位置 t 只能看到 t 之前的内容。
""",
    ),
    (
        "rope",
        "旋转位置编码",
        """
RoPE 把位置写进向量的相位，旋转矩阵是正交的，因此内积只依赖相对位移。
高频维度在训练长度内会转很多圈，外推到更长序列时相位混叠，性能下降。
位置插值把长序列的位置压缩回训练范围，代价是分辨率整体下降。
NTK-aware 缩放只放大低频维度的波长，保留高频的局部区分能力。
""",
    ),
    (
        "moe",
        "混合专家",
        """
MoE 让每个 token 只激活少数专家，参数量按总参数算，计算量按激活参数算。
路由容易塌缩：某个专家被选得越多学得越快，形成正反馈。
负载均衡损失等于专家数乘以分配比例与平均路由概率的乘积之和，均匀时取最小值。
容量因子限制每个专家能接收的 token 数，超出部分会被丢弃。
""",
    ),
    (
        "kv",
        "KV cache 与推理显存",
        """
推理显存由三部分组成：权重、KV cache、激活。权重是常数，KV cache 随上下文线性增长。
每个 token 的 KV 大小等于 2 乘以层数乘以隐藏维度乘以字节数。
7B 模型每 token 约 512 KB，128K 上下文就是 69 GB，比权重还大。
分页注意力把 KV cache 切成固定大小的块，碎片率从 60% 降到 4%。
""",
    ),
    (
        "quant",
        "量化",
        """
量化用低比特表示权重，把 70B 模型从 140 GB 压到 35 GB，从两张卡变成一张卡。
INT8 几乎无损，INT4 需要分组量化才能把困惑度上升控制在 0.3 以内。
GPTQ 用 Hessian 近似逐层量化，AWQ 按激活重要性保护少数通道。
KV cache 量化到 INT8 通常安全，INT4 会明显掉点。
""",
    ),
    (
        "rag",
        "检索增强生成",
        """
RAG 先用检索找到证据，再基于证据生成答案。切分决定召回上限，
块太小语义不完整，块太大噪声多，甜点区在 256 到 512 token。
混合检索用 RRF 融合稀疏与稠密两路排名，k 取 60 时头部差距被压平。
重排用交叉编码器对召回结果精排，通常带来 5 到 15 个点的精度提升。
""",
    ),
    (
        "agent",
        "Agent 与误差累积",
        """
Agent 由模型策略、工具、记忆和循环控制组成。单步成功率 0.95 时，
20 步任务的整体成功率只有 0.95 的 20 次方，约 36%。
所以工程上要缩短链路、加检查点、允许重试。
反思机制在没有外部验证器时会让性能下降，因为模型无法可靠判断自己错在哪。
""",
    ),
    (
        "scaling",
        "缩放定律",
        """
损失随参数量与数据量呈幂律下降。给定算力时，参数与数据应同比增加，
计算最优配比约为每参数 20 个 token。Chinchilla 之前所有模型都训练不足。
总成本最优则不同：推理量大时，小模型多喂数据更划算，
临界点由训练多花的算力除以每 token 省下的算力决定。
""",
    ),
]

# 查询 → 金标文档
QUERIES = [
    ("注意力为什么要除以根号 d", ["attention"]),
    ("MoE 的路由塌缩是怎么回事", ["moe"]),
    ("128K 上下文的 KV cache 有多大", ["kv"]),
    ("量化到 4 bit 有什么代价", ["quant"]),
    ("RAG 的切分块多大合适", ["rag"]),
    ("agent 的误差为什么会累积", ["agent"]),
    ("参数和数据应该怎么配比", ["scaling"]),
    ("外推到更长序列为什么失效", ["rope"]),
]


# ---------------------------------------------------------------- 基础工具


def tokenize(text: str) -> list[str]:
    """中文按字、英文按词，全部小写"""
    text = text.lower()
    zh = re.findall(r"[\u4e00-\u9fa5]", text)
    en = re.findall(r"[a-z0-9]+", text)
    return zh + en


def chunk_docs(size: int = 60, overlap: int = 10) -> list[dict]:
    """按固定窗口切分，带重叠"""
    chunks = []
    for doc_id, title, body in DOCS:
        toks = tokenize(body)
        step = size - overlap
        for start in range(0, max(1, len(toks) - overlap), step):
            piece = toks[start : start + size]
            if len(piece) < 8:
                continue
            chunks.append({"doc": doc_id, "title": title, "tokens": piece})
    return chunks


def bm25(
    query: list[str], chunks: list[dict], k1: float = 1.5, b: float = 0.75
) -> list[tuple[int, float]]:
    """标准 BM25"""
    n = len(chunks)
    avg_len = sum(len(c["tokens"]) for c in chunks) / n
    df = Counter()
    for c in chunks:
        df.update(set(c["tokens"]))

    scores = []
    for i, c in enumerate(chunks):
        tf = Counter(c["tokens"])
        s = 0.0
        for q in set(query):
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            denom = tf[q] + k1 * (1 - b + b * len(c["tokens"]) / avg_len)
            s += idf * tf[q] * (k1 + 1) / denom
        scores.append((i, s))
    return sorted(scores, key=lambda x: x[1], reverse=True)


class LSA:
    """TF-IDF + SVD 的稠密检索（相当于一个没训练过的嵌入模型）"""

    def __init__(self, chunks: list[dict], dim: int = 32):
        vocab = sorted({t for c in chunks for t in c["tokens"]})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n = len(chunks)
        df = Counter()
        for c in chunks:
            df.update(set(c["tokens"]))
        self.idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1 for t in vocab])

        X = np.zeros((n, len(vocab)))
        for i, c in enumerate(chunks):
            for t, cnt in Counter(c["tokens"]).items():
                X[i, self.vocab[t]] = (1 + math.log(cnt)) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1, norms)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(dim, S.size)
        self.doc_vecs = U[:, :k] * S[:k]
        self.proj = Vt[:k].T

    def score(self, query: list[str]) -> list[tuple[int, float]]:
        q = np.zeros(len(self.vocab))
        for t, cnt in Counter(query).items():
            if t in self.vocab:
                q[self.vocab[t]] = (1 + math.log(cnt)) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(q)
        if norm == 0:
            return [(i, 0.0) for i in range(len(self.doc_vecs))]
        qv = (q / norm) @ self.proj
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        d = self.doc_vecs / (np.linalg.norm(self.doc_vecs, axis=1, keepdims=True) + 1e-9)
        sims = d @ qv
        return sorted(((i, float(s)) for i, s in enumerate(sims)), key=lambda x: x[1], reverse=True)


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """倒数排名融合"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(
    query: list[str],
    chunks: list[dict],
    candidates: list[int],
    dense_map: dict[int, float],
    top_k: int = 5,
) -> list[int]:
    """极简重排：稠密相似度 + 词面覆盖 + 标题命中（模拟交叉编码器的作用）"""
    q = set(query)
    scored = []
    for i in candidates:
        c = chunks[i]
        overlap = len(q & set(c["tokens"])) / (len(q) + 1e-9)
        title_hit = len(q & set(tokenize(c["title"]))) * 0.3
        scored.append((i, 0.6 * dense_map.get(i, 0.0) + 0.4 * overlap + title_hit))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:top_k]]


def generate(query: list[str], chunks: list[dict], picked: list[int], confident: bool) -> str:
    """抽取式生成。

    检索置信时只复述上下文；不置信时"补"一句上下文里没有的话——
    模拟真实系统在证据不足时编内容的行为，用来验证忠实度指标能否抓到。
    """
    best = max(picked, key=lambda i: len(set(query) & set(chunks[i]["tokens"])))
    body = "".join(chunks[best]["tokens"][:40])
    answer = f"{body}。[1]"
    if not confident:
        answer += "此外，该方法在所有任务上都优于人类专家。"
    return answer


# ---------------------------------------------------------------- 评估


def evaluate(chunks: list[dict], retrieved: list[int], answer: str, gold: list[str]) -> dict:
    gold_chunks = {i for i, c in enumerate(chunks) if c["doc"] in gold}
    hit = set(retrieved) & gold_chunks
    ctx_precision = len(hit) / max(1, len(retrieved))
    ctx_recall = len(hit) / max(1, len(gold_chunks))

    # 忠实度：答案的每一句是否能在上下文里找到足够词面支撑
    ctx_tokens = set()
    for i in retrieved:
        ctx_tokens |= set(chunks[i]["tokens"])
    sentences = [s for s in re.split(r"[。！？]", answer) if len(s.strip()) > 4]
    supported = 0
    for s in sentences:
        toks = set(tokenize(s))
        if len(toks & ctx_tokens) / max(1, len(toks)) > 0.6:
            supported += 1
    faithfulness = supported / max(1, len(sentences))

    # 答案相关性：答案与查询的词面重合
    q = set(tokenize(" ".join(body for doc_id, _, body in DOCS if doc_id in gold)))
    a = set(tokenize(answer))
    relevance = len(a & q) / max(1, len(q)) if q else 0.0

    return {
        "ctx_precision": ctx_precision,
        "ctx_recall": ctx_recall,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
    }


# ---------------------------------------------------------------- 主流程


def run(mode: str, use_rerank: bool) -> dict:
    chunks = chunk_docs()
    lsa = LSA(chunks)
    totals = Counter()

    for query_text, gold in QUERIES:
        q = tokenize(query_text)
        sparse = [i for i, _ in bm25(q, chunks)[:20]]
        dense = [i for i, _ in lsa.score(q)[:20]]

        if mode == "bm25":
            retrieved = sparse[:3]
            confident = bm25(q, chunks)[0][1] > 6.0
        elif mode == "dense":
            retrieved = dense[:3]
            confident = lsa.score(q)[0][1] > 0.55
        else:
            fused = [i for i, _ in rrf([sparse, dense])][:3]
            retrieved = fused
            confident = lsa.score(q)[0][1] > 0.55

        if use_rerank:
            pool = list(dict.fromkeys(sparse + dense))[:20]
            dense_map = dict(lsa.score(q))
            retrieved = rerank(q, chunks, pool, dense_map, top_k=3)

        answer = generate(q, chunks, retrieved, confident)
        m = evaluate(chunks, retrieved, answer, gold)
        for k, v in m.items():
            totals[k] += v

    n = len(QUERIES)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    chunks = chunk_docs()
    print(f"语料 {len(DOCS)} 篇 → 切成 {len(chunks)} 块\n")

    configs = [
        ("BM25 单路", "bm25", False),
        ("向量单路", "dense", False),
        ("混合（RRF）", "hybrid", False),
        ("混合 + 重排", "hybrid", True),
    ]

    header = f"{'配置':<14}{'上下文精度':>10}{'上下文召回':>10}{'忠实度':>10}{'答案相关性':>10}"
    print(header)
    print("-" * 62)
    for name, mode, rr in configs:
        m = run(mode, rr)
        print(
            f"{name:<14}{m['ctx_precision']:>10.3f}{m['ctx_recall']:>10.3f}"
            f"{m['faithfulness']:>10.3f}{m['answer_relevance']:>10.3f}"
        )

    print("\n注：检索不置信时会混入一句上下文没有的话，忠实度随之下降。")

    print("\n混合检索的逐条结果：")
    lsa = LSA(chunks)
    for query_text, gold in QUERIES:
        q = tokenize(query_text)
        fused = [
            i
            for i, _ in rrf(
                [[i for i, _ in bm25(q, chunks)[:20]], [i for i, _ in lsa.score(q)[:20]]]
            )
        ][:3]
        hit = sum(1 for i in fused if chunks[i]["doc"] in gold)
        print(f"  {query_text:<22} 命中 {hit}/3   金标 {gold[0]}")


if __name__ == "__main__":
    main()
