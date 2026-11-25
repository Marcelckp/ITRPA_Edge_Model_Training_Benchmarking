"""FinQA: official train/dev/test JSON files from the FinQA repository."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import read_records, seeded_sample, write_jsonl, write_manifest

log = logging.getLogger(__name__)


def normalise(rec: dict[str, Any]) -> dict[str, Any]:
    qa = rec.get("qa", {})
    gold = qa.get("exe_ans", qa.get("answer", ""))
    return {
        "query_id": str(rec["id"]),
        "question": qa.get("question", ""),
        "pre_text": list(rec.get("pre_text", [])),
        "post_text": list(rec.get("post_text", [])),
        "table": [list(map(str, row)) for row in rec.get("table", [])],
        "gold": str(gold),
        "gold_text": str(qa.get("answer", "")),
        "gold_inds": sorted((qa.get("gold_inds") or {}).keys()),
    }


def prepare(out_dir: str | Path, *, train: str | None = None, dev: str | None = None, test: str | None = None,
            n_test: int = 500, n_train: int | None = None, seed: int = 42, **_: Any) -> dict[str, Any]:
    if not (train and dev and test):
        raise ValueError("finqa needs --train, --dev and --test (official FinQA JSON files)")
    out = Path(out_dir) / "finqa"
    test_rows = seeded_sample([normalise(r) for r in read_records(test)], n_test, seed)
    train_rows = seeded_sample([normalise(r) for r in read_records(train)], n_train, seed)
    dev_rows = sorted((normalise(r) for r in read_records(dev)), key=lambda r: r["query_id"])
    counts = {"test": write_jsonl(out / "test.jsonl", test_rows), "train": write_jsonl(out / "train.jsonl", train_rows),
              "val": write_jsonl(out / "val.jsonl", dev_rows)}
    info = {"dataset": "finqa", "seed": seed, "counts": counts}
    write_manifest(out, info)
    log.info("finqa prepared: %s", info)
    return info
