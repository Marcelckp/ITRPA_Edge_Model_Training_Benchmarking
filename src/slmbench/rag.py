"""BM25 retrieval: PubMedQA corpus-level abstracts and FinQA within-document sentences/table rows.

Rankings are precomputed once (``build-index``) so every device sees identical retrieved contexts.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

from .datasets import read_jsonl, write_jsonl
from .prompts import finqa_units

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what which "
    "with does did do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _bm25_class():
    from rank_bm25 import BM25Okapi  # lazy: [rag] extra

    class LuceneBM25(BM25Okapi):
        """Okapi BM25 with the Lucene IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

        The classic IDF is zero or negative for terms in half or more of the documents, which happens often
        in FinQA's short per-document unit lists and would make ranking arbitrary.
        """

        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    return LuceneBM25


class BM25Index:
    def __init__(self, doc_ids: Sequence[str], docs: Sequence[str], tokens: Sequence[Sequence[str]] | None = None):
        BM25Okapi = _bm25_class()
        self.doc_ids = list(doc_ids)
        self.docs = list(docs)
        self.tokens = [list(t) for t in tokens] if tokens is not None else [tokenize(d) for d in docs]
        # rank_bm25 divides by zero on an all-empty corpus.
        safe = [t if t else ["__empty__"] for t in self.tokens]
        self._bm25 = BM25Okapi(safe)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Top-k (doc_id, text, score); ties broken by corpus position for determinism."""
        if k <= 0 or not self.doc_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        k = min(k, len(order))
        return [(self.doc_ids[i], self.docs[i], float(scores[i])) for i in order[:k]]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            for i, d, t in zip(self.doc_ids, self.docs, self.tokens):
                fh.write(json.dumps({"doc_id": i, "text": d, "tokens": t}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        ids, docs, toks = [], [], []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                ids.append(r["doc_id"])
                docs.append(r["text"])
                toks.append(r["tokens"])
        return cls(ids, docs, toks)


def build_pubmedqa(data_dir: str | Path, max_k: int) -> dict[str, Any]:
    root = Path(data_dir) / "pubmedqa"
    corpus = read_jsonl(root / "corpus.jsonl")
    t0 = time.perf_counter()
    index = BM25Index([d["doc_id"] for d in corpus], [d["text"] for d in corpus])
    index.save(root / "bm25.jsonl.gz")
    test = read_jsonl(root / "test.jsonl")
    rows, q_times = [], []
    for it in test:
        q0 = time.perf_counter()
        hits = index.search(it["question"], max_k)
        q_times.append(time.perf_counter() - q0)
        rows.append({"query_id": it["query_id"], "doc_ids": [h[0] for h in hits], "texts": [h[1] for h in hits],
                     "gold_units": [it["query_id"]]})
    write_jsonl(root / "retrieval.jsonl", rows)
    info = {"dataset": "pubmedqa", "corpus_docs": len(index), "queries": len(rows), "max_k": max_k,
            "build_s": time.perf_counter() - t0,
            "mean_query_ms": 1000 * sum(q_times) / len(q_times) if q_times else None}
    log.info("pubmedqa index: %s", info)
    return info


def build_finqa(data_dir: str | Path, max_k: int) -> dict[str, Any]:
    root = Path(data_dir) / "finqa"
    test = read_jsonl(root / "test.jsonl")
    rows, q_times = [], []
    for it in test:
        units = finqa_units(it)
        q0 = time.perf_counter()
        index = BM25Index([u[0] for u in units], [u[1] for u in units])
        hits = index.search(it["question"], max_k)
        q_times.append(time.perf_counter() - q0)
        rows.append({"query_id": it["query_id"], "doc_ids": [h[0] for h in hits], "texts": [h[1] for h in hits],
                     "gold_units": list(it.get("gold_inds", []))})
    write_jsonl(root / "retrieval.jsonl", rows)
    info = {"dataset": "finqa", "queries": len(rows), "max_k": max_k,
            "mean_query_ms": 1000 * sum(q_times) / len(q_times) if q_times else None}
    log.info("finqa index: %s", info)
    return info


def build(dataset: str, data_dir: str | Path, max_k: int) -> dict[str, Any]:
    if dataset == "pubmedqa":
        return build_pubmedqa(data_dir, max_k)
    if dataset == "finqa":
        return build_finqa(data_dir, max_k)
    raise ValueError(f"no retrieval for dataset {dataset}")


def load_rankings(data_dir: str | Path, dataset: str) -> dict[str, dict[str, Any]]:
    path = Path(data_dir) / dataset / "retrieval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `slmbench build-index --dataset {dataset}` first")
    return {r["query_id"]: r for r in read_jsonl(path)}


def top_k(ranking: dict[str, Any], k: int) -> tuple[list[str], int | None]:
    """Top-k texts and the retrieval hit flag (1 if every gold unit is in the top k; None without gold)."""
    if k > len(ranking["doc_ids"]):
        log.debug("k=%d exceeds precomputed depth %d", k, len(ranking["doc_ids"]))
    ids = ranking["doc_ids"][:max(k, 0)]
    gold = ranking.get("gold_units") or []
    hit = None if not gold else int(all(g in ids for g in gold))
    return ranking["texts"][:max(k, 0)], hit


def recall_at_k(ranking: dict[str, Any], k: int) -> float | None:
    gold = ranking.get("gold_units") or []
    if not gold:
        return None
    ids = set(ranking["doc_ids"][:k])
    return sum(1 for g in gold if g in ids) / len(gold)
