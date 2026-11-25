import json
import subprocess
import sys

import pytest

from slmbench import db
from slmbench import train as T
from slmbench.config import ConfigError
from slmbench.export import export, gguf_commands, record_artifact

from .conftest import ROOT, SMALL_MATRIX


def test_train_dry_run_does_not_import_torch(tmp_path):
    code = ("import sys, json; from slmbench.cli import main; "
            f"rc = main(['train', '--matrix', '{SMALL_MATRIX}', '--db', '{tmp_path}/r.db', '--model', 'toy-a', "
            "'--strategy', 'qlora', '--dataset', 'pubmedqa', '--lr', '1e-4', '--dry-run']); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; sys.exit(rc)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan["quantization"]["quant_type"] == "nf4" and plan["lora"]["r"] == 16
    assert plan["gradient_accumulation_steps"] == 4 and plan["optimizer"] == "adamw_torch"


def test_lr_must_be_in_grid(small_matrix):
    with pytest.raises(ConfigError):
        T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, 5e-4, "data")
    with pytest.raises(ConfigError):
        T.make_plan(small_matrix, "toy-a", "base", "pubmedqa", 42, 1e-4, "data")


def test_same_optimiser_and_checkpointing_across_strategies(small_matrix):
    plans = [T.make_plan(small_matrix, "toy-a", s, "finqa", 42, lr, "data")
             for s, lr in (("fft", 1e-5), ("lora", 1e-4), ("qlora", 2e-4))]
    assert len({(p["optimizer"], p["gradient_checkpointing"], p["effective_batch_size"]) for p in plans}) == 1


def test_lr_selection_marks_best_validation(tmp_path, small_matrix):
    conn = db.connect(tmp_path / "r.db")
    for lr, val in ((1e-4, 0.6), (2e-4, 0.7)):
        plan = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 42, lr, "data")
        db.insert_training_run(conn, T.mock_training_row(plan, val))
    assert T.select_learning_rates(conn, 42) == []  # mock rows are ignored without allow_mock
    chosen = T.select_learning_rates(conn, 42, allow_mock=True)
    assert len(chosen) == 1 and chosen[0]["learning_rate"] == 2e-4
    assert T.selected_lr(conn, "toy-a", "lora", "pubmedqa") == 2e-4
    plan = T.make_plan(small_matrix, "toy-a", "lora", "pubmedqa", 43, 2e-4, "data")
    db.insert_training_run(conn, T.mock_training_row(plan))
    T.select_learning_rates(conn, 42, allow_mock=True)
    assert conn.execute("SELECT selected FROM training_runs WHERE seed=43").fetchone()[0] == 0


def test_gguf_commands_force_f16():
    cmds = gguf_commands("/m/merged", {"F16": "/o/f16.gguf", "Q8_0": "/o/q8.gguf", "Q4_K_M": "/o/q4.gguf"},
                         "/lc", python="py")
    assert cmds[0][:2] == ["py", "/lc/convert_hf_to_gguf.py"]
    assert cmds[0][cmds[0].index("--outtype") + 1] == "f16"
    assert cmds[1] == ["/lc/build/bin/llama-quantize", "/o/f16.gguf", "/o/q8.gguf", "Q8_0"]
    assert cmds[2][-1] == "Q4_K_M"


def test_export_dry_run_paths(small_matrix):
    plan = export(small_matrix, None, "toy-a", "lora", "finqa", "/t/lr0.0001-s42", "/lc", dry_run=True)
    assert plan["outputs"]["Q4_K_M"].endswith("toy-a-lora-finqa-Q4_K_M.gguf")
    base = export(small_matrix, None, "toy-a", "base", "", None, "/lc", dry_run=True)
    assert base["outputs"]["F16"].endswith("toy-a-base-F16.gguf") and base["merged_dir"] == "toy/a"
    with pytest.raises(ValueError):
        export(small_matrix, None, "toy-a", "lora", "finqa", None, "/lc", dry_run=True)


def test_record_artifact_sha(tmp_path):
    conn = db.connect(tmp_path / "r.db")
    f = tmp_path / "x.gguf"
    f.write_bytes(b"abc")
    row = record_artifact(conn, "m", "lora", "finqa", "Q8_0", f)
    assert row["size_bytes"] == 3
    assert row["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert conn.execute("SELECT dataset FROM artifacts").fetchone()[0] == "finqa"
