from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slmbench import config as C  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
RAW = FIX / "raw"
SMALL_MATRIX = FIX / "matrix_small.yaml"


@pytest.fixture
def small_matrix() -> dict:
    return C.load_matrix(SMALL_MATRIX)


@pytest.fixture
def full_matrix() -> dict:
    return C.load_matrix(C.DEFAULT_MATRIX)


@pytest.fixture
def prepared_data(tmp_path: Path, small_matrix: dict) -> Path:
    from slmbench.datasets import preparer
    from slmbench.rag import build

    out = tmp_path / "data"
    preparer("pubmedqa")(out, labeled=str(RAW / "pubmedqa_labeled.jsonl"),
                         artificial=str(RAW / "pubmedqa_artificial.jsonl"),
                         test_ids_file=str(RAW / "pubmedqa_test_ids.json"), n_test=12, n_train=10, seed=42)
    preparer("finqa")(out, train=str(RAW / "finqa_train.json"), dev=str(RAW / "finqa_dev.json"),
                      test=str(RAW / "finqa_test.json"), n_test=10, seed=42)
    preparer("arc_easy")(out, test=str(RAW / "arc_easy_test.jsonl"), n_test=10, seed=42)
    build("pubmedqa", out, 10)
    build("finqa", out, 10)
    return out
