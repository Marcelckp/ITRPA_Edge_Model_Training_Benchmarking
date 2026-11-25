"""End-to-end with the mock backend: prepare -> index -> train(mock) -> run -> analyze -> report -> advise."""

import json
import sqlite3

import pandas as pd
import pytest
from PIL import Image

from slmbench import MOCK_WATERMARK
from slmbench import config as C
from slmbench.cli import main

from .conftest import RAW, SMALL_MATRIX


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data, dbp = root / "data", root / "results.db"
    m = ["--matrix", str(SMALL_MATRIX)]
    assert main(["prepare-data", *m, "--dataset", "pubmedqa", "--out", str(data),
                 "--labeled", str(RAW / "pubmedqa_labeled.jsonl"), "--artificial", str(RAW / "pubmedqa_artificial.jsonl"),
                 "--test-ids", str(RAW / "pubmedqa_test_ids.json")]) == 0
    assert main(["prepare-data", *m, "--dataset", "finqa", "--out", str(data), "--train", str(RAW / "finqa_train.json"),
                 "--dev", str(RAW / "finqa_dev.json"), "--test", str(RAW / "finqa_test.json")]) == 0
    assert main(["prepare-data", *m, "--dataset", "arc_easy", "--out", str(data),
                 "--test", str(RAW / "arc_easy_test.jsonl")]) == 0
    for ds in ("pubmedqa", "finqa"):
        assert main(["build-index", *m, "--dataset", ds, "--data-dir", str(data)]) == 0
    matrix = C.load_matrix(SMALL_MATRIX)
    for job in C.training_plan(matrix, "selection"):
        assert main(["train", *m, "--db", str(dbp), "--model", job["model"], "--strategy", job["strategy"],
                     "--dataset", job["dataset"], "--seed", "42", "--lr", str(job["learning_rate"]), "--mock"]) == 0
    assert main(["train", *m, "--db", str(dbp), "--select", "--allow-mock"]) == 0
    for job in C.training_plan(matrix, "variability"):
        assert main(["train", *m, "--db", str(dbp), "--model", job["model"], "--strategy", job["strategy"],
                     "--dataset", job["dataset"], "--seed", str(job["seed"]), "--lr", "selected", "--mock"]) == 0
    common = [*m, "--db", str(dbp), "--data-dir", str(data), "--backend", "mock", "--idle-s", "0.002"]
    for plan in ("accuracy", "retention", "ref-gold"):
        assert main(["run", *common, "--plan", plan, "--device", "gpu-workstation", "--repeats", "1"]) == 0
    assert main(["run", *common, "--plan", "edge", "--device", "edge-pi5"]) == 0
    return root, data, dbp, m


def test_runs_recorded_as_mock(pipeline):
    _, _, dbp, _ = pipeline
    conn = sqlite3.connect(dbp)
    matrix = C.load_matrix(SMALL_MATRIX)
    expected = sum(c.n_items for p in ("accuracy", "retention", "ref-gold", "edge")
                   for c in C.expand(matrix, [p], is_mock=True))
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == expected
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE is_mock=0").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM training_runs WHERE selected=1").fetchone()[0] == 12
    n_edge = len(C.expand(matrix, ["edge"]))
    assert conn.execute("SELECT COUNT(*) FROM energy_blocks b JOIN configs c USING(config_id) WHERE c.tier='edge'"
                        ).fetchone()[0] == 3 * n_edge


def test_analyze_outputs(pipeline):
    root, _, dbp, m = pipeline
    out = root / "analysis"
    assert main(["analyze", *m, "--db", str(dbp), "--out", str(out)]) == 3
    assert not out.exists()
    assert main(["analyze", *m, "--db", str(dbp), "--out", str(out), "--allow-mock"]) == 0
    for csv in out.glob("*.csv"):
        frame = pd.read_csv(csv)
        assert frame.columns[0] == "watermark" and (frame.empty or (frame.watermark == MOCK_WATERMARK).all()), csv
    ht = pd.read_csv(out / "hypothesis_tests.csv")
    assert {"H1a", "H1b", "H2a", "H2b", "H3", "SQ2", "SQ3"} <= set(ht.hypothesis)
    assert ht.decision.notna().all()
    assert set(ht[ht.hypothesis == "H1a"].decision) == {"supported"}
    mem = ht[ht.family.str.startswith("H1b-mem")]
    assert len(mem) and mem.test.eq("friedman").all() and mem.p.notna().all()
    h1b_energy = ht[ht.family.str.startswith("H1b-energy")]
    assert len(h1b_energy) and h1b_energy.decision.isin(["supported", "not supported"]).all()
    assert (ht[ht.hypothesis == "H2a"].decision == "not testable").any()  # toy n is below the floor rule
    parity = pd.read_csv(out / "accuracy_parity.csv")
    assert len(parity) and parity.agreement.notna().all() and "flag_rerun_500" in parity
    be = pd.read_csv(out / "break_even.csv")
    assert len(be) and (be.break_even_queries.notna() | be.reason.str.startswith("undefined")).all()
    cost = pd.read_csv(out / "cost.csv")
    assert {"zar_per_correct", "kgco2e_per_1000_queries"} <= set(cost.columns)
    cs = pd.read_csv(out / "config_summary.csv")
    assert cs.parse_fail_rate.notna().all() and "macro_f1" in cs


def test_report_refuses_mock_then_watermarks(pipeline):
    root, _, dbp, m = pipeline
    out = root / "generated"
    assert main(["report", *m, "--db", str(dbp), "--out", str(out)]) == 3
    assert not (out / "hypothesis_decisions.md").exists()
    assert main(["report", *m, "--db", str(dbp), "--out", str(out), "--allow-mock"]) == 0
    mds = list(out.rglob("*.md"))
    pngs = list((out / "figures").glob("*.png"))
    assert len(mds) >= 10 and len(pngs) >= 4
    for p in mds:
        assert MOCK_WATERMARK in p.read_text(), p
    for p in pngs:
        assert Image.open(p).text.get("Description") == MOCK_WATERMARK, p
    hd = (out / "hypothesis_decisions.md").read_text()
    assert "Newcombe" in hd and "not testable" in hd and "Page's L" in hd
    assert json.loads((out / "manifest.json").read_text())["is_mock"] is True


def test_advise_html(pipeline):
    root, _, dbp, m = pipeline
    html_path = root / "advisor.html"
    args = ["advise", *m, "--db", str(dbp), "--device", "edge-pi5", "--monthly-queries", "50000",
            "--latency-budget-s", "5", "--min-accuracy", "0.0", "--dataset", "pubmedqa", "--out", str(html_path)]
    assert main(args) == 3 and not html_path.exists()
    assert main([*args, "--allow-mock"]) == 0
    html = html_path.read_text()
    assert MOCK_WATERMARK in html and "Feasible configurations" in html and "<script" not in html
    assert "Recommended:" in html


def test_estimate_and_merge(pipeline, capsys):
    root, _, dbp, m = pipeline
    capsys.readouterr()
    assert main(["estimate", *m, "--db", str(dbp), "--plan", "edge", "--device", "edge-pi5", "--mock"]) == 0
    out = json.loads(capsys.readouterr().out)
    proj = out["projection"]["edge-pi5"]
    assert proj["measured"] == out["counts"]["configs"] and proj["hours"] > 0
    merged = root / "merged.db"
    assert main(["merge-db", "--into", str(merged), str(dbp)]) == 0
    a = sqlite3.connect(dbp).execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert sqlite3.connect(merged).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == a


def test_energy_block_cli(tmp_path, monkeypatch):
    dbp = tmp_path / "e.db"
    assert main(["energy-block", "--db", str(dbp), "start", "--config-id", "cfg", "--device", "edge-pi5",
                 "--meter-mwh", "1000"]) == 0
    assert main(["energy-block", "--db", str(dbp), "stop", "--block-id", "cfg-r0-manual", "--meter-mwh", "1010",
                 "--n-queries", "100"]) == 0
    row = sqlite3.connect(dbp).execute("SELECT energy_j, n_queries FROM energy_blocks").fetchone()
    assert row == (36.0, 100)
