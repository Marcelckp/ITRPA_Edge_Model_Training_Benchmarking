import json
from collections import Counter

import pytest

from slmbench import prompts as P
from slmbench import rag
from slmbench.datasets import (class_balanced_sample, edge_subset, load_split, read_jsonl, seeded_sample,
                               stratified_sample)


def test_pubmedqa_split_rules(prepared_data):
    test = load_split(prepared_data, "pubmedqa", "test")
    val = load_split(prepared_data, "pubmedqa", "val")
    train = load_split(prepared_data, "pubmedqa", "train")
    corpus = read_jsonl(prepared_data / "pubmedqa" / "corpus.jsonl")
    test_ids = {r["query_id"] for r in test}
    val_ids = {r["query_id"] for r in val}
    assert len(test) == 12 and len(val) == 12 and not test_ids & val_ids
    assert not ({r["query_id"] for r in train} & (test_ids | val_ids))  # pqa_labeled PMIDs removed
    ids = [d["doc_id"] for d in corpus]
    assert len(ids) == len(set(ids))  # deduplicated by PMID
    assert test_ids <= set(ids)
    man = json.loads((prepared_data / "pubmedqa" / "manifest.json").read_text())
    assert man["artificial_removed_overlap"] == 2 and man["official_test_split"]


def test_pubmedqa_requires_official_split(tmp_path):
    from slmbench.datasets.pubmedqa import prepare

    from .conftest import RAW

    with pytest.raises(ValueError):
        prepare(tmp_path, labeled=str(RAW / "pubmedqa_labeled.jsonl"),
                artificial=str(RAW / "pubmedqa_artificial.jsonl"))


def test_finqa_and_arc_prepared(prepared_data):
    fin = load_split(prepared_data, "finqa", "test")
    assert len(fin) == 10 and any(r["gold"] == "yes" for r in fin)
    assert len(load_split(prepared_data, "finqa", "val")) == 4
    arc = load_split(prepared_data, "arc_easy", "test")
    assert all(r["gold"] in "ABCD" and [c["label"] for c in r["choices"]] == list("ABCD") for r in arc)


def _items(n, labels=("yes", "no", "maybe"), weights=(5, 3, 2)):
    out, i = [], 0
    for lab, w in zip(labels, weights):
        for _ in range(w * n):
            out.append({"query_id": f"q{i:04d}", "gold": lab})
            i += 1
    return out


def test_sampling_is_seeded_and_order_independent():
    items = _items(10)
    a = seeded_sample(items, 20, 42)
    b = seeded_sample(list(reversed(items)), 20, 42)
    assert a == b and len(a) == 20
    assert seeded_sample(items, None, 42) == sorted(items, key=lambda r: r["query_id"])


def test_stratified_subset_proportional_and_identical():
    items = _items(10)
    s1 = stratified_sample(items, 10, 42)
    assert Counter(r["gold"] for r in s1) == {"yes": 5, "no": 3, "maybe": 2}
    assert s1 == edge_subset(items, 10, "stratified", 42)


def test_class_balanced_redistributes_shortfall():
    items = _items(1, weights=(10, 6, 1))
    s = class_balanced_sample(items, 12, 42)
    c = Counter(r["gold"] for r in s)
    assert len(s) == 12 and c["maybe"] == 1 and abs(c["yes"] - c["no"]) <= 1


def test_bm25_ranking_and_k_handling():
    idx = rag.BM25Index(["d1", "d2", "d3"], ["zorblex reduces toy cough", "quantium raises toy rash", "unrelated text"])
    hits = idx.search("does zorblex reduce cough", 2)
    assert hits[0][0] == "d1" and len(hits) == 2
    assert idx.search("zorblex", 0) == []
    assert len(idx.search("zorblex", 50)) == 3


def test_bm25_save_load_roundtrip(tmp_path):
    idx = rag.BM25Index(["a", "b"], ["alpha beta", "gamma delta"])
    idx.save(tmp_path / "i.jsonl.gz")
    again = rag.BM25Index.load(tmp_path / "i.jsonl.gz")
    assert again.search("gamma", 1)[0][0] == "b"


def test_rankings_hit_and_recall(prepared_data):
    r = rag.load_rankings(prepared_data, "pubmedqa")
    one = next(iter(r.values()))
    texts, hit = rag.top_k(one, 3)
    assert len(texts) == 3 and hit in (0, 1)
    assert rag.recall_at_k(one, 10) in (0.0, 1.0)
    fin = next(iter(rag.load_rankings(prepared_data, "finqa").values()))
    assert set(fin["gold_units"]) == {"table_1", "text_1"}
    assert all(u.startswith(("text_", "table_")) for u in fin["doc_ids"])


def test_finqa_units_and_truncation():
    item = {"question": "q?", "pre_text": ["p0", "p1"], "post_text": ["z0"], "table": [["h", "a"], ["rev", "1"]]}
    ids = [u for u, _ in P.finqa_units(item)]
    assert ids == ["text_0", "text_1", "table_0", "table_1", "text_2"]
    long = {**item, "pre_text": ["word " * 3000] * 3}
    p = P.build_prompt("finqa", long, "full", context_tokens=4096, max_new_tokens=32)
    assert p.truncated
    short = P.build_prompt("finqa", item, "full")
    assert not short.truncated and "rev: a = 1" in short.messages[1]["content"]


def test_prompt_modes_and_render():
    item = {"question": "Is it?", "contexts": ["gold abstract"], "gold": "yes"}
    closed = P.build_prompt("pubmedqa", item, "closed")
    gold = P.build_prompt("pubmedqa", item, "gold")
    assert "Context" not in closed.messages[1]["content"] and "gold abstract" in gold.messages[1]["content"]
    rendered = P.render(closed.messages)
    assert rendered.startswith("System:") and rendered.endswith("Assistant:")

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return "<T>" + "|".join(m["role"] for m in msgs)

    assert P.render(closed.messages, Tok()) == "<T>system|user"
    with pytest.raises(ValueError):
        P.build_prompt("pubmedqa", item, "weird")
