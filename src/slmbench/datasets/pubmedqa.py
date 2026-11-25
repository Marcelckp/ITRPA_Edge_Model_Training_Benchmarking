"""PubMedQA: official ori_pqal/ori_pqaa JSON or Hugging Face `qiaojin/PubMedQA` records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import class_balanced_sample, read_records, seeded_sample, write_jsonl, write_manifest

log = logging.getLogger(__name__)

LABELS = ("yes", "no", "maybe")
HF_PATH = "qiaojin/PubMedQA"


def normalise(rec: dict[str, Any]) -> dict[str, Any]:
    if "QUESTION" in rec:
        pmid = str(rec.get("_key") or rec.get("pubid"))
        return {"query_id": pmid, "question": rec["QUESTION"], "contexts": list(rec.get("CONTEXTS", [])),
                "gold": str(rec.get("final_decision", "")).lower()}
    ctx = rec.get("context", {})
    contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else list(ctx or [])
    return {"query_id": str(rec["pubid"]), "question": rec["question"], "contexts": list(contexts),
            "gold": str(rec.get("final_decision", "")).lower()}


def load_hf(config: str) -> list[dict[str, Any]]:
    import datasets  # lazy: only on the workstation

    ds = datasets.load_dataset(HF_PATH, config, split="train")
    return [dict(r) for r in ds]


def load_test_ids(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k) for k in (data.keys() if isinstance(data, dict) else data)}


def prepare(out_dir: str | Path, *, labeled: str | None = None, artificial: str | None = None,
            test_ids_file: str | None = None, n_test: int = 500, n_train: int | None = 10000,
            class_balanced: bool = True, seed: int = 42, use_hf: bool = False,
            allow_unofficial_split: bool = False, **_: Any) -> dict[str, Any]:
    out = Path(out_dir) / "pubmedqa"
    lab_raw = read_records(labeled) if labeled else (load_hf("pqa_labeled") if use_hf else None)
    art_raw = read_records(artificial) if artificial else (load_hf("pqa_artificial") if use_hf else None)
    if lab_raw is None or art_raw is None:
        raise ValueError("pubmedqa needs --labeled and --artificial files (or --hf)")
    lab = [normalise(r) for r in lab_raw]
    art = [normalise(r) for r in art_raw]
    lab_ids = {r["query_id"] for r in lab}

    if test_ids_file:
        test_ids = load_test_ids(test_ids_file)
        missing = test_ids - lab_ids
        if missing:
            raise ValueError(f"{len(missing)} test PMIDs not found in pqa_labeled")
    elif allow_unofficial_split:
        log.warning("no test_ids_file: using a seeded 50/50 split of pqa_labeled (NOT the official split)")
        test_ids = {r["query_id"] for r in seeded_sample(lab, len(lab) // 2, seed)}
    else:
        raise ValueError("pubmedqa needs test_ids_file (official test_ground_truth.json) "
                         "or --allow-unofficial-split")
    test = seeded_sample([r for r in lab if r["query_id"] in test_ids], n_test, seed)
    val = [r for r in sorted(lab, key=lambda r: r["query_id"]) if r["query_id"] not in test_ids]

    art_clean = [r for r in art if r["query_id"] not in lab_ids]
    removed = len(art) - len(art_clean)
    sampler = class_balanced_sample if class_balanced else seeded_sample
    train = sampler(art_clean, n_train, seed)

    corpus: dict[str, dict[str, str]] = {}
    for r in lab + art:
        corpus.setdefault(r["query_id"], {"doc_id": r["query_id"], "text": " ".join(r["contexts"])})
    docs = [corpus[k] for k in sorted(corpus)]

    counts = {"test": write_jsonl(out / "test.jsonl", test), "val": write_jsonl(out / "val.jsonl", val),
              "train": write_jsonl(out / "train.jsonl", train), "corpus": write_jsonl(out / "corpus.jsonl", docs)}
    info = {"dataset": "pubmedqa", "seed": seed, "counts": counts, "artificial_removed_overlap": removed,
            "official_test_split": bool(test_ids_file),
            "train_label_counts": {lab_: sum(1 for r in train if r["gold"] == lab_) for lab_ in LABELS}}
    write_manifest(out, info)
    log.info("pubmedqa prepared: %s", info)
    return info
