"""Dataset preparation: normalise raw files into a local JSONL cache with seeded, documented sampling."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

DATASETS = ("pubmedqa", "finqa", "arc_easy")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a .jsonl file, a JSON list, or a JSON object keyed by id (PubMedQA official format)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [{"_key": k, **v} if isinstance(v, dict) else {"_key": k, "value": v} for k, v in data.items()]
    raise ValueError(f"{p}: unsupported JSON layout")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sorted(items: list[dict], key: str = "query_id") -> list[dict]:
    return sorted(items, key=lambda r: str(r[key]))


def seeded_sample(items: list[dict], n: int | None, seed: int, key: str = "query_id") -> list[dict]:
    """Deterministic sample independent of input order; returns all items when n is None or too large."""
    pool = _sorted(items, key)
    if n is None or n >= len(pool):
        return pool
    picked = random.Random(seed).sample(pool, n)
    return _sorted(picked, key)


def stratified_sample(items: list[dict], n: int | None, seed: int, label: str = "gold",
                      key: str = "query_id") -> list[dict]:
    """Proportional allocation per label (largest remainder), seeded within each stratum."""
    pool = _sorted(items, key)
    if n is None or n >= len(pool):
        return pool
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in pool:
        groups[str(it[label])].append(it)
    labels = sorted(groups)
    exact = {lab: n * len(groups[lab]) / len(pool) for lab in labels}
    alloc = {lab: int(exact[lab]) for lab in labels}
    rest = n - sum(alloc.values())
    for lab in sorted(labels, key=lambda x: (-(exact[x] - alloc[x]), x))[:rest]:
        alloc[lab] += 1
    rng = random.Random(seed)
    out: list[dict] = []
    for lab in labels:
        out.extend(rng.sample(groups[lab], alloc[lab]))
    return _sorted(out, key)


def class_balanced_sample(items: list[dict], n: int | None, seed: int, label: str = "gold",
                          key: str = "query_id") -> list[dict]:
    """Equal count per class; a class with too few items gives its shortfall to the others."""
    pool = _sorted(items, key)
    if n is None:
        n = len(pool)
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in pool:
        groups[str(it[label])].append(it)
    labels = sorted(groups)
    alloc = {lab: 0 for lab in labels}
    remaining = min(n, len(pool))
    open_labels = list(labels)
    while remaining > 0 and open_labels:
        share = max(1, remaining // len(open_labels))
        for lab in list(open_labels):
            take = min(share, len(groups[lab]) - alloc[lab], remaining)
            alloc[lab] += take
            remaining -= take
            if alloc[lab] >= len(groups[lab]):
                open_labels.remove(lab)
            if remaining == 0:
                break
    rng = random.Random(seed)
    out: list[dict] = []
    for lab in labels:
        out.extend(rng.sample(groups[lab], alloc[lab]))
    return _sorted(out, key)


def edge_subset(items: list[dict], n: int | None, strategy: str, seed: int) -> list[dict]:
    if strategy == "stratified":
        return stratified_sample(items, n, seed)
    return seeded_sample(items, n, seed)


def write_manifest(out_dir: Path, info: dict[str, Any]) -> None:
    files = {p.name: file_sha256(p) for p in sorted(out_dir.glob("*.jsonl"))}
    (out_dir / "manifest.json").write_text(json.dumps({**info, "sha256": files}, indent=2, sort_keys=True))


def load_split(data_dir: str | Path, dataset: str, split: str = "test") -> list[dict[str, Any]]:
    path = Path(data_dir) / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `slmbench prepare-data --dataset {dataset}` first")
    return read_jsonl(path)


def preparer(dataset: str) -> Callable[..., dict[str, Any]]:
    if dataset == "pubmedqa":
        from .pubmedqa import prepare
    elif dataset == "finqa":
        from .finqa import prepare
    elif dataset == "arc_easy":
        from .arc import prepare
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return prepare
