"""GPU training kit (TRAINING-GPU.md): training matrix, pinned stack, notebook commands and an offline mock run."""

import ast
import json
import re
import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest

from slmbench import config as C
from slmbench import train as T
from slmbench.cli import build_parser
from slmbench.datasets import preparer

from .conftest import RAW, ROOT

GPU = C.DEFAULT_CONFIG_DIR / "matrix-train-gpu.yaml"
NOTEBOOK = ROOT.parent / "training" / "kaggle_train.ipynb"
needs_notebook = pytest.mark.skipif(not NOTEBOOK.exists(), reason="training/ is not shipped with the prototype")
MEMORY_KEYS = {"optimizer", "per_device_batch_size"}


def _with_llama(path: Path, tmp_path: Path) -> dict:
    p = tmp_path / "m.yaml"
    p.write_text(path.read_text().replace("#llama ", ""))
    return C.load_matrix(p)


def _counts(m: dict) -> dict:
    return {p: (len(cs), C.count_runs(cs)["queries"]) for p in m["plans"] for cs in [C.expand(m, [p])]}


def test_gpu_matrix_is_design_recipe_except_memory_profile(tmp_path, full_matrix):
    gpu = _with_llama(GPU, tmp_path)
    for key in ("chat_template_kwargs", "context_tokens", "prompt_overhead_tokens", "max_new_tokens", "datasets"):
        assert gpu[key] == full_matrix[key], key
    assert {m["id"]: m for m in gpu["models"]} == {m["id"]: m for m in full_matrix["models"]}
    a, b = gpu["training"], full_matrix["training"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == MEMORY_KEYS
    assert (a["optimizer"], a["per_device_batch_size"], a["effective_batch_size"]) == ("adamw_bnb_8bit", 1, 16)
    plans = [T.make_plan(gpu, m, s, d, 42, float(gpu["training"]["lr_grid"][s][0]), "data")
             for m in ("qwen2.5-0.5b", "llama3.2-1b") for s in ("fft", "lora", "qlora") for d in ("pubmedqa", "finqa")]
    assert {(p["optimizer"], p["per_device_batch_size"], p["gradient_accumulation_steps"]) for p in plans} == \
        {("adamw_bnb_8bit", 1, 16)}
    labels = {T.precision_recipe(p, bf16_supported=False)["label"].split()[0:3][-1] for p in plans}
    assert labels == {"fp16-autocast"}  # T4 and P100 have no native bf16


def test_gpu_matrix_keeps_pilot_ids_and_paths(tmp_path):
    gpu, pilot = C.load_matrix(GPU), C.load_matrix(C.DEFAULT_CONFIG_DIR / "matrix-pilot.yaml")
    ids = lambda m: {c.config_id for c in C.expand(m) if c.strategy in ("base", "rag")}  # noqa: E731
    assert ids(gpu) == ids(pilot)
    assert _counts(gpu) == {"accuracy": (84, 42000), "retention": (42, 12600), "edge": (84, 8400),
                            "ref-gold": (6, 3000)}
    assert len(C.training_plan(gpu, "selection")) == 24 and len(C.training_plan(gpu, "variability")) == 24
    edge = C.expand(gpu, ["edge"])
    assert all(c.model_path.startswith("/cache/models-gpu/") for c in edge if c.strategy in ("fft", "lora", "qlora"))
    assert all(c.model_path.startswith("/cache/models/") for c in edge if c.strategy in ("base", "rag"))
    assert C.trained_model_dir(gpu, "qwen2.5-0.5b", "fft", "finqa", 1e-5, 42).startswith("/cache/models-gpu-ckpt/")
    assert C.merged_model_dir(gpu, "qwen2.5-0.5b", "fft", "finqa").startswith("/cache/models-gpu-ckpt/")
    llama = _with_llama(GPU, tmp_path)
    assert _counts(llama)["accuracy"][0] == 126 and len(C.training_plan(llama, "selection")) == 36


def test_constraints_pin_every_gpu_dependency():
    pins = {}
    for line in (ROOT / "constraints-gpu.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            pins[name.lower()] = version
    # Same transformers/torch as llama.cpp b12532's conversion requirements, so one venv trains and converts.
    assert pins["transformers"] == "4.57.6" and pins["torch"] == "2.11.0"
    assert pins["bitsandbytes"] == "0.49.2" and pins["peft"] and pins["accelerate"]
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for dep in extras["gpu"] + extras["rag"] + extras["analysis"]:
        assert re.split(r"[<>=!~ ]", dep)[0].lower() in pins, dep


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def _code(nb: dict) -> list[tuple[str, list]]:
    return [("".join(c["source"]), c["metadata"].get("tags", [])) for c in nb["cells"] if c["cell_type"] == "code"]


@needs_notebook
def test_notebook_structure_and_cli_flags():
    nb = _notebook()
    assert nb["nbformat"] == 4 and len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
    code = _code(nb)
    params = [src for src, tags in code if "parameters" in tags]
    assert params[0].strip().splitlines()[-1].startswith("MODEL =")
    assert params[1].strip().splitlines()[-1].startswith("DATASET =")
    subs = {a.dest: a for a in build_parser()._actions if a.dest == "cmd"}["cmd"].choices
    used: dict[str, set] = {}

    def record(sub, nodes):
        flags = {n.value for top in nodes for n in ast.walk(top)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("--")}
        used.setdefault(sub, set()).update(flags)

    for src, _ in code:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("slm", "slm_check"):
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant):
                    record(first.value, node.args)
            if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant) \
                    and node.elts[0].value in subs:
                record(node.elts[0].value, node.elts)
            if isinstance(node, ast.FunctionDef) and node.name == "common_args":
                record("train", list(ast.walk(node)))
    assert {"env", "prepare-data", "train", "export-gguf", "merge-db"} <= set(used)
    for sub, flags in used.items():
        assert sub in subs, sub
        known = set(subs[sub]._option_string_actions)
        assert flags <= known, (sub, sorted(flags - known))


@needs_notebook
def test_notebook_never_exposes_the_hf_token():
    code = [src for src, _ in _code(_notebook())]
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", "\n".join(code))
    leaks = []
    for src in code:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Call, ast.JoinedStr)):
                inner = node.args + [k.value for k in node.keywords] if isinstance(node, ast.Call) else node.values
                if any(isinstance(n, ast.Name) and n.id == "HF_TOKEN" for a in inner for n in ast.walk(a)):
                    leaks.append(ast.unparse(node))
    assert leaks == []
    assert 'BASE_ENV["HF_TOKEN"] = HF_TOKEN' in "\n".join(code)


def _execute(label, tmp_path, data, **over):
    """Run the notebook's code cells in-process in mock mode, parameters overridden after the last parameter cell."""
    params = dict(MODEL=["toy-a"], DATASET=["pubmedqa"], MOCK=True, PROTOTYPE_SRC=str(ROOT),
                  MATRIX="tests/fixtures/matrix_small.yaml", OUT_DIR=str(tmp_path / label / "out"),
                  SCRATCH_DIR=str(tmp_path / label / "scratch"), PREPARED_DATA=str(data), PRIOR_OUTPUTS=[],
                  LINK_DEV_HOST_PATHS=False, PROBE=False, SESSION_LABEL=label,
                  PHASES=["selection", "export", "variability"])
    params.update(over)
    code = _code(_notebook())
    last = max(i for i, (_, tags) in enumerate(code) if "parameters" in tags)
    ns: dict = {"__name__": "__main__"}
    for i, (src, _) in enumerate(code):
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
        if i == last:
            ns.update(params)
    return ns


@pytest.fixture
def toy_data(tmp_path):
    out = tmp_path / "data"
    preparer("pubmedqa")(out, labeled=str(RAW / "pubmedqa_labeled.jsonl"),
                         artificial=str(RAW / "pubmedqa_artificial.jsonl"),
                         test_ids_file=str(RAW / "pubmedqa_test_ids.json"), n_test=12, n_train=10, seed=42)
    return out


@needs_notebook
def test_notebook_mock_run_resumes_per_run(tmp_path, toy_data):
    s1 = _execute("mock-s01", tmp_path, toy_data, STRATEGIES=["fft"], PHASES=["selection"])
    assert len(s1["DONE_JOBS"]) == 2 and not s1["MOCK_EXPORTED"]
    out1 = tmp_path / "mock-s01" / "out"
    meta = json.loads((out1 / "session.json").read_text())
    assert meta["clean_exit"] and meta["mock"] and meta["gpu_name"] == "mock-gpu"
    assert not list((out1 / "results").glob("*.db-wal"))
    s2 = _execute("mock-s02", tmp_path, toy_data, PRIOR_OUTPUTS=[str(out1)])
    assert len(s2["DONE_JOBS"]) == 10  # fft seed 42 came from s01; lora/qlora 2 + 2 each, 3 x 2 variability seeds
    assert s2["MOCK_EXPORTED"] == {("toy-a", s, "pubmedqa") for s in ("fft", "lora", "qlora")}
    s3 = _execute("mock-s03", tmp_path, toy_data, PRIOR_OUTPUTS=[str(out1), str(tmp_path / "mock-s02" / "out")])
    assert s3["DONE_JOBS"] == [] and s3["FAILED"] == []
    db3 = tmp_path / "mock-s03" / "out" / "results" / "train-mock-s03.db"
    con = sqlite3.connect(db3)
    rows = con.execute("SELECT strategy, seed, COUNT(*) FROM training_runs GROUP BY strategy, seed").fetchall()
    assert sorted(rows) == sorted((s, seed, n) for s in ("fft", "lora", "qlora") for seed, n in ((42, 2), (43, 1), (44, 1)))
    assert con.execute("SELECT COUNT(*) FROM training_runs WHERE selected=1").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM environment WHERE key LIKE 'mock-s0_-gpu0/%'").fetchone()[0] > 0
    sums = (tmp_path / "mock-s03" / "out" / "results" / "SHA256SUMS-mock-s03").read_text()
    assert "train-mock-s03.db" in sums


@needs_notebook
def test_notebook_mock_budget_stops_cleanly(tmp_path, toy_data):
    ns = _execute("mock-short", tmp_path, toy_data, SESSION_BUDGET_H=0.1)
    assert ns["DONE_JOBS"] == [] and len(set(ns["INCOMPLETE"])) == 3
    meta = json.loads((tmp_path / "mock-short" / "out" / "session.json").read_text())
    assert meta["clean_exit"] and len(meta["incomplete_cells"]) == 3


@needs_notebook
def test_notebook_refuses_mixed_gpu_campaign(tmp_path, toy_data):
    _execute("mock-a", tmp_path, toy_data, STRATEGIES=["fft"], PHASES=["selection"])
    out = tmp_path / "mock-a" / "out"
    meta = json.loads((out / "session.json").read_text())
    (out / "session.json").write_text(json.dumps({**meta, "gpu_name": "Tesla P100-PCIE-16GB"}))
    with pytest.raises(SystemExit, match="one GPU model"):
        _execute("mock-b", tmp_path, toy_data, PRIOR_OUTPUTS=[str(out)])


def test_eval_adapted_script_uses_its_own_matrix_db_and_report():
    script = ROOT / "scripts" / "train-gpu" / "eval_adapted.sh"
    text = script.read_text()
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert 'MATRIX="configs/matrix-train-gpu.yaml"' in text and "results/pilot-adapted-gpu.db" in text
    assert "generated-pilot-gpu" in text and "dissertation/generated " not in text + " "
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/../pilot/common.sh"' in text
