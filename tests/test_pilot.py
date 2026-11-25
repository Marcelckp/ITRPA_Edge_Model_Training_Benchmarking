"""Container-emulated edge pilot: matrix counts, tier profiles, resumable export, preflight filter, labelled report."""

import sqlite3
import subprocess
from pathlib import Path

import pytest

from slmbench import config as C
from slmbench import env as E
from slmbench import export as X
from slmbench import preflight as P
from slmbench.cli import main

from .conftest import ROOT

PILOT = C.DEFAULT_CONFIG_DIR / "matrix-pilot.yaml"
SCRIPTS = ROOT / "scripts" / "pilot"


@pytest.fixture
def pilot() -> dict:
    return C.load_matrix(PILOT)


def _counts(m: dict) -> dict:
    return {p: (len(cs), C.count_runs(cs)["queries"]) for p in m["plans"] for cs in [C.expand(m, [p])]}


def test_pilot_counts(pilot):
    # Per model per domain dataset: base + rag K3/K5/K10 = 4 variants x 3 precisions = 12.
    assert _counts(pilot) == {"accuracy": (48, 24000), "retention": (6, 1800), "ref-gold": (6, 3000),
                              "edge": (48, 4800)}
    assert sum(n for n, _ in _counts(pilot).values()) == 108
    assert {c.strategy for c in C.expand(pilot)} == {"base", "rag"}
    assert C.adapted_model_count(pilot) == 0 and C.training_plan(pilot, "selection") == []


def test_pilot_tiers_and_paths(pilot):
    acc = C.expand(pilot, ["accuracy", "retention", "ref-gold"])
    edge = C.expand(pilot, ["edge"])
    assert {(c.tier, c.device) for c in acc} == {("cpu", "cpu-reference")}
    assert {(c.tier, c.device, c.n_items) for c in edge} == {("edge", "edge-container-4c8g", 100)}
    assert all(c.model_path.startswith("/cache/models/") and c.model_path.endswith(".gguf") for c in acc + edge)
    assert {c.context_mode for c in C.expand(pilot, ["retention"])} == {"retention:none"}


def test_pilot_llama_block_parses_when_enabled(tmp_path):
    text = PILOT.read_text().replace("#llama ", "")
    p = tmp_path / "matrix-pilot-llama.yaml"
    p.write_text(text)
    m = C.load_matrix(p)
    assert [x["id"] for x in m["models"]] == ["qwen2.5-0.5b", "smollm2-360m", "llama3.2-1b"]
    assert {k: v[0] for k, v in _counts(m).items()} == {"accuracy": 72, "retention": 9, "ref-gold": 9, "edge": 72}


def test_pilot_device_profiles():
    ref, edge = C.load_device("cpu-reference"), C.load_device("edge-container-4c8g")
    assert (ref["tier"], edge["tier"]) == ("cpu", "edge")
    for dev, threads in ((ref, "16"), (edge, "4")):
        cmd = dev["backends"]["llamacpp"]["server_cmd"]
        assert cmd[cmd.index("-t") + 1] == threads and cmd[cmd.index("-c") + 1] == "4096"
        assert cmd[cmd.index("--load-mode") + 1] == "mmap+mlock" and cmd[cmd.index("--cache-ram") + 1] == "0"
        assert "--jinja" in cmd
        assert dev["telemetry"]["power"] == [] and dev["telemetry"]["memory"] == ["rss"]
        assert "block_energy" not in dev["telemetry"]


def test_pilot_scripts_limits_and_outputs():
    common = (SCRIPTS / "common.sh").read_text()
    assert '--cpuset-cpus "$EDGE_CPUS" --cpus 4 --memory 8g --memory-swap 8g' in common
    assert 'EDGE_CPUS="${EDGE_CPUS:-16-19}"' in common and 'REF_CPUS="${REF_CPUS:-0-15}"' in common
    assert "HF_TOKEN=" not in common.replace('"${HF_TOKEN:-}"', "")
    report = (SCRIPTS / "07_report.sh").read_text()
    assert "generated-pilot" in report and "dissertation/generated " not in report + " "
    for s in sorted(SCRIPTS.glob("*.sh")):
        assert subprocess.run(["bash", "-n", str(s)], capture_output=True).returncode == 0, s.name


def _fake_tools(calls):
    def run(cmd, check):
        calls.append(cmd)
        out = cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[2]
        assert out.endswith(".part")
        Path(out).write_bytes(b"gguf:" + cmd[-1].encode())
    return run


def test_export_precision_and_skip_existing(tmp_path, monkeypatch, small_matrix):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "toy" / "a").mkdir(parents=True)  # stands in for a downloaded HF snapshot
    calls: list = []
    monkeypatch.setattr(X.subprocess, "run", _fake_tools(calls))
    from slmbench import db

    conn = db.connect(tmp_path / "r.db")
    models = tmp_path / "models"
    plan = X.export(small_matrix, conn, "toy-a", "base", "", None, "/lc", str(models), precisions=("Q4_K_M",),
                    skip_existing=True)
    assert [X._cmd_precision(c) for c in calls] == ["F16", "Q4_K_M"]
    assert list(plan["outputs"]) == ["Q4_K_M"]
    assert not list(models.rglob("*.part"))
    assert [r[0] for r in conn.execute("SELECT precision FROM artifacts")] == ["Q4_K_M"]

    calls.clear()
    X.export(small_matrix, conn, "toy-a", "base", "", None, "/lc", str(models), precisions=("Q4_K_M",),
             skip_existing=True)
    assert calls == []
    assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1

    X.export(small_matrix, conn, "toy-a", "base", "", None, "/lc", str(models), skip_existing=True)
    assert [X._cmd_precision(c) for c in calls] == ["Q8_0"]
    assert sorted(r[0] for r in conn.execute("SELECT precision FROM artifacts")) == ["F16", "Q4_K_M", "Q8_0"]


def test_export_interrupted_leaves_no_final_file(tmp_path, monkeypatch, small_matrix):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "toy" / "a").mkdir(parents=True)

    def boom(cmd, check):
        out = cmd[cmd.index("--outfile") + 1]
        Path(out).write_bytes(b"half")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(X.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        X.export(small_matrix, None, "toy-a", "base", "", None, "/lc", str(tmp_path / "m"), skip_existing=True)
    assert not list((tmp_path / "m").rglob("*.gguf"))


def test_export_cli_precision_flag(tmp_path, capsys):
    rc = main(["export-gguf", "--matrix", str(PILOT), "--db", str(tmp_path / "x.db"), "--model", "smollm2-360m",
               "--strategy", "base", "--precision", "Q4_K_M", "--models-dir", "/cache/models", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "smollm2-360m-base-Q4_K_M.gguf" in out and '"Q8_0"' not in out


def test_preflight_cpu_tier_checks_libraries_and_filter(monkeypatch, tmp_path, capsys):
    libs = [c.name for c in P.check_libraries({"tier": "cpu"}, P.Host(version=lambda n: "99.0", python=(3, 11)))]
    assert libs == ["python", "libs[edge]", "libs[rag]", "libs[analysis]"]
    seen = {}

    def fake(dev, cfgs, *a, **k):
        seen["paths"] = {c.model_path for c in cfgs}
        return [P.Check("x", P.PASS, "ok")]

    monkeypatch.setattr(P, "run_preflight", fake)
    assert main(["preflight", "--matrix", str(PILOT), "--device", "edge-container-4c8g", "--db",
                 str(tmp_path / "s.db"), "--filter", "model=smollm2-360m", "--filter", "precision=Q4_K_M"]) == 0
    assert seen["paths"] == {"/cache/models/smollm2-360m/base/smollm2-360m-base-Q4_K_M.gguf"}


def test_env_commit_falls_back_to_image_variable(monkeypatch):
    monkeypatch.setattr(E, "_run", lambda cmd: "fatal: not a git repository" if cmd[0] == "git" else None)
    monkeypatch.setenv("LLAMA_CPP_COMMIT", "08c252e221759e0f69e80f0450b295c92d2344e1")
    monkeypatch.setenv("LLAMA_CPP_TAG", "b12532")
    env = E.collect("edge-container-4c8g", "/opt/llama.cpp")
    assert env["edge-container-4c8g/llama_cpp_commit"] == "08c252e221759e0f69e80f0450b295c92d2344e1"
    assert env["edge-container-4c8g/llama_cpp_tag"] == "b12532"


def test_pilot_mock_report_is_labelled(tmp_path, prepared_data):
    dbp = tmp_path / "pilot.db"
    common = ["--matrix", str(PILOT), "--db", str(dbp), "--data-dir", str(prepared_data), "--backend", "mock"]
    assert main(["run", *common, "--device", "cpu-reference", "--plan", "accuracy", "--plan", "retention",
                 "--plan", "ref-gold"]) == 0
    assert main(["run", *common, "--device", "edge-container-4c8g", "--plan", "edge"]) == 0
    conn = sqlite3.connect(dbp)
    devices = {r[0] for r in conn.execute("SELECT DISTINCT device FROM configs")}
    assert devices == {"cpu-reference", "edge-container-4c8g"}
    assert conn.execute("SELECT COUNT(*) FROM energy_blocks b JOIN configs c USING(config_id) "
                        "WHERE c.tier='edge'").fetchone()[0] == 48  # energy_repeats 1
    out = tmp_path / "generated-pilot"
    assert main(["report", "--matrix", str(PILOT), "--db", str(dbp), "--out", str(out), "--allow-mock"]) == 0
    edge_md = (out / "tables" / "edge_latency_memory.md").read_text()
    assert "container-emulated" in edge_md and "Not Raspberry Pi results" in edge_md
    assert "Energy not measured: no energy counter" in (out / "tables" / "energy_blocks.md").read_text()
    decisions = (out / "hypothesis_decisions.md").read_text()
    assert "| H1b | H1b-energy | all |" in decisions and "not measured" in decisions
