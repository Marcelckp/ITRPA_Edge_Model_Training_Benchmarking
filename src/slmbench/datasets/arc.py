"""ARC-Easy retention probe: official ARC JSONL or Hugging Face `allenai/ai2_arc` records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import read_records, seeded_sample, write_jsonl, write_manifest

log = logging.getLogger(__name__)

HF_PATH = "allenai/ai2_arc"
LETTERS = "ABCDE"


def _letter(label: str, idx: int) -> str:
    label = str(label).strip().upper()
    if label in LETTERS:
        return label
    if label.isdigit() and 1 <= int(label) <= len(LETTERS):
        return LETTERS[int(label) - 1]
    return LETTERS[idx]


def normalise(rec: dict[str, Any]) -> dict[str, Any]:
    q = rec["question"]
    if isinstance(q, dict):
        stem, raw = q["stem"], q["choices"]
        choices = [(c["label"], c["text"]) for c in raw]
    else:
        stem = q
        ch = rec["choices"]
        choices = list(zip(ch["label"], ch["text"]))
    mapping = {str(lab): _letter(lab, i) for i, (lab, _) in enumerate(choices)}
    return {"query_id": str(rec["id"]), "question": stem,
            "choices": [{"label": mapping[str(lab)], "text": txt} for lab, txt in choices],
            "gold": mapping.get(str(rec["answerKey"]), _letter(rec["answerKey"], 0))}


def load_hf() -> list[dict[str, Any]]:
    import datasets  # lazy

    return [dict(r) for r in datasets.load_dataset(HF_PATH, "ARC-Easy", split="test")]


def prepare(out_dir: str | Path, *, test: str | None = None, n_test: int = 300, seed: int = 42,
            use_hf: bool = False, **_: Any) -> dict[str, Any]:
    raw = read_records(test) if test else (load_hf() if use_hf else None)
    if raw is None:
        raise ValueError("arc_easy needs --test file (or --hf)")
    out = Path(out_dir) / "arc_easy"
    rows = seeded_sample([normalise(r) for r in raw], n_test, seed)
    info = {"dataset": "arc_easy", "seed": seed, "counts": {"test": write_jsonl(out / "test.jsonl", rows)}}
    write_manifest(out, info)
    log.info("arc_easy prepared: %s", info)
    return info
