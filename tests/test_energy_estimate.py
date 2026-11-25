import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from slmbench import db
from slmbench import energy_estimate as E
from slmbench.config import DEFAULT_TARIFF, load_tariff
from slmbench.telemetry.base import TelemetrySession
from slmbench.telemetry.cgroup import (CgroupSource, HostSample, find_cgroup, parse_v2_cpu_ns, read_cgroup,
                                       read_host_csv, window_usage)

from .conftest import ROOT

CID = "526ded4fd278a9e5d73a7d2145e2ba5f7db20edd4f0f0862ba465bb88f6d9f70"


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _v1_root(root: Path, rel: str = "") -> Path:
    base = Path(rel)
    _write(root / "cpuacct" / base / "cpuacct.usage", "19152165545333\n")
    _write(root / "cpu" / base / "cpu.stat", "nr_periods 10\nnr_throttled 2\nthrottled_time 500\n")
    _write(root / "memory" / base / "memory.usage_in_bytes", "321277952\n")
    _write(root / "memory" / base / "memory.max_usage_in_bytes", "324661248\n")
    _write(root / "memory" / base / "memory.stat", "cache 3784704\nrss 307777536\ntotal_rss 307777536\ntotal_cache 3784704\n")
    return root


def _v2_root(root: Path, rel: str = "") -> Path:
    _write(root / "cgroup.controllers", "cpu memory\n")
    d = root / rel if rel else root
    _write(d / "cpu.stat", "usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\nnr_throttled 3\nthrottled_usec 40\n")
    _write(d / "memory.current", "104857600\n")
    _write(d / "memory.peak", "209715200\n")
    _write(d / "memory.stat", "anon 90000000\nfile 14857600\n")
    return root


def test_cgroup_v1_inside_container(tmp_path):
    root = _v1_root(tmp_path)
    p = find_cgroup(root, f"9:cpu,cpuacct:/docker/{CID}\n3:memory:/docker/{CID}\n")
    assert p.version == 1
    r = read_cgroup(p)
    assert r.cpu_ns == 19152165545333 and r.mem_bytes == 321277952 and r.mem_peak_bytes == 324661248
    src = CgroupSource(root, f"9:cpu,cpuacct:/docker/{CID}\n")
    assert src.available() and src.cpu_s() == pytest.approx(19152.165545333)
    assert src.read().mem_mb == pytest.approx(321277952 / 2**20)


def test_cgroup_v1_host_path_preferred(tmp_path):
    root = _v1_root(tmp_path, f"docker/{CID}")
    p = find_cgroup(root, f"9:cpu,cpuacct:/docker/{CID}\n3:memory:/docker/{CID}\n")
    assert p.version == 1 and CID in str(p.cpu)


def test_cgroup_v2(tmp_path):
    assert parse_v2_cpu_ns("usage_usec 7\n") == 7000
    root = _v2_root(tmp_path, f"system.slice/docker-{CID}.scope")
    p = find_cgroup(root, f"0::/system.slice/docker-{CID}.scope\n")
    assert p.version == 2
    r = read_cgroup(p)
    assert r.cpu_ns == 2_500_000_000 and r.mem_bytes == 104857600 and r.mem_peak_bytes == 209715200
    assert find_cgroup(_v2_root(tmp_path / "ns"), "0::/\n").version == 2
    assert find_cgroup(tmp_path / "empty", "") is None


def _logger():
    spec = importlib.util.spec_from_file_location("cgroup_logger", ROOT / "scripts" / "pilot" / "cgroup_logger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_host_logger_parses_v1_and_v2(tmp_path):
    lg = _logger()
    v, d = lg.cgroup_dirs(_v1_root(tmp_path / "a", f"docker/{CID}"), CID)
    s = lg.sample(v, d)
    assert v == 1 and s["cpu_usage_ns"] == 19152165545333 and s["mem_rss_bytes"] == 307777536
    assert s["nr_throttled"] == 2 and s["throttled_ns"] == 500
    v, d = lg.cgroup_dirs(_v2_root(tmp_path / "b", f"system.slice/docker-{CID}.scope"), CID)
    s = lg.sample(v, d)
    assert v == 2 and s["cpu_usage_ns"] == 2_500_000_000 and s["mem_peak_bytes"] == 209715200
    assert s["throttled_ns"] == 40_000 and s["mem_cache_bytes"] == 14857600
    assert lg.cgroup_dirs(tmp_path / "none", CID) is None


def _samples(cid="c1", n=11, rate=2.0, t0=1000.0):
    return [HostSample(t0 + i, int(i * rate * 1e9), (100 + i) * 2**20, cid) for i in range(n)]


def test_window_interpolation():
    s = _samples()
    u = window_usage(s, 1002.5, 1005.5)
    assert u.cpu_s == pytest.approx(6.0) and u.wall_s == pytest.approx(3.0)
    assert u.peak_mem_mb == pytest.approx(106.0)
    assert window_usage(s, 1002.2, 1002.7).cpu_s == pytest.approx(1.0)
    assert window_usage(s, 999.0, 1002.0).cpu_s is None
    assert window_usage(s, 1009.5, 1011.0).cpu_s is None
    mixed = _samples("a", 5) + [HostSample(1005 + i, i, None, "b") for i in range(5)]
    assert window_usage(mixed, 1003.0, 1006.0).cpu_s is None


def _pilot_db(tmp_path: Path):
    conn = db.connect(tmp_path / "p.db")
    db.upsert_config(conn, {"config_id": "cfgA", "model": "m", "strategy": "base", "precision": "Q4_K_M",
                            "engine": "llamacpp", "tier": "edge", "device": "edge-container-4c8g", "dataset": "pubmedqa",
                            "context_mode": "closed", "k": 0, "model_path": "/x", "seed": 42, "temperature": 0.0,
                            "max_new_tokens": 16, "is_mock": 0})
    from datetime import datetime, timezone

    def iso(t):
        return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")
    for i, (t, dur) in enumerate(((1001.0, 2.0), (1004.0, 3.0))):
        db.insert_run(conn, {"config_id": "cfgA", "query_id": f"q{i}", "block_id": "b1", "started_at": iso(t),
                             "e2e_latency_s": dur, "correct": 1, "is_mock": 0})
    db.upsert_energy_block(conn, {"block_id": "b1", "config_id": "cfgA", "device": "edge-container-4c8g",
                                  "repeat_index": 0, "started_at": iso(1001.0), "ended_at": iso(1007.0),
                                  "n_queries": 2, "source": None})
    csv_path = tmp_path / "edge-cgroup.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utc", "unix_s", "container_id", "cgroup_version", "cpu_usage_ns", "mem_usage_bytes",
                    "mem_peak_bytes", "mem_rss_bytes", "mem_cache_bytes", "nr_throttled", "throttled_ns"])
        for s in _samples(n=12, rate=4.0):
            w.writerow([iso(s.t), f"{s.t:.3f}", s.container_id, 1, s.cpu_ns, s.mem_bytes, "", "", "", 0, 0])
    return conn, csv_path


def test_join_model_and_labelling(tmp_path):
    conn, csv_path = _pilot_db(tmp_path)
    assert len(read_host_csv(csv_path)) == 12
    counts = E.import_host_csv(conn, csv_path, "edge-container-4c8g")
    assert counts == {"query": 2, "block": 1, "uncovered": 0}
    blk = conn.execute("SELECT * FROM cpu_usage WHERE scope='block'").fetchone()
    assert blk["cpu_s"] == pytest.approx(24.0) and blk["wall_s"] == pytest.approx(6.0) and blk["n_queries"] == 2
    q1 = conn.execute("SELECT cpu_s FROM cpu_usage WHERE ref_id='cfgA:q1'").fetchone()[0]
    assert q1 == pytest.approx(12.0)

    model = E.load_model()
    assert E.compute(conn, model) == 3
    est = conn.execute("SELECT * FROM energy_estimates WHERE scope='block'").fetchone()
    mem_gb = blk["peak_mem_mb"] / 1024
    assert est["energy_basis"] == "estimated:green-algorithms" and "not metered" in est["method_note"]
    assert est["energy_j_central"] == pytest.approx(24.0 * 6.875 + 6.0 * mem_gb * 0.3725)
    assert est["energy_j_low"] < est["energy_j_central"] < est["energy_j_high"]
    # Estimates never leak into measured-energy columns.
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE energy_j IS NOT NULL").fetchone()[0] == 0
    assert conn.execute("SELECT energy_j FROM energy_blocks").fetchone()[0] is None

    tariff = load_tariff(DEFAULT_TARIFF)
    tables = E.summary_tables(conn, tariff)
    inf = tables["estimated_energy_inference"]
    assert inf.j_per_query_central.iloc[0] == pytest.approx(est["energy_j_central"] / 2)
    kwh_1000 = inf.j_per_query_central.iloc[0] * 1000 / 3.6e6
    assert inf.zar_per_1000_queries_central.iloc[0] == pytest.approx(kwh_1000 * 4.14)
    assert inf.kgco2e_per_1000_queries_central.iloc[0] == pytest.approx(kwh_1000 * 0.960)
    assert (inf.energy_basis == "estimated:green-algorithms").all() and inf.label.str.contains("not metered").all()
    files = E.write_outputs(tables, model, tariff, tmp_path / "out", caption="Pilot, container-emulated edge")
    md = (tmp_path / "out" / "estimated_energy.md").read_text()
    assert "estimated from CPU time, not metered" in md and "Green Algorithms" in md
    with open(next(f for f in files if f.endswith("inference.csv"))) as fh:
        row = next(csv.DictReader(fh))
    assert row["energy_basis"] == "estimated:green-algorithms" and "not metered" in row["label"]


def test_model_maths_and_validation(tmp_path):
    m = E.EnergyModel({"low": 5.0, "central": 6.0, "high": 7.0}, 0.3725, pue=1.0)
    assert m.energy_j(100.0, 50.0, 8.0) == pytest.approx(100 * 6.0 + 50 * 8 * 0.3725)
    assert m.energy_j(100.0, 50.0, None, "low") == pytest.approx(500.0)
    m2 = E.EnergyModel({"low": 5.0, "central": 6.0, "high": 7.0}, 0.3725, pue=1.5, threads_per_core=2.0)
    assert m2.energy_j(100.0, 0.0, 0.0) == pytest.approx(1.5 * 50 * 6.0)
    d = E.load_model()
    assert d.p_core_w == {"low": 6.25, "central": 6.875, "high": 7.2917} and d.p_mem_w_per_gb == 0.3725 and d.pue == 1.0
    bad = tmp_path / "bad.yaml"
    bad.write_text("p_core_w: {low: 1, central: 2, high: 3}\np_mem_w_per_gb: 0.3\nlabel: measured energy\n")
    with pytest.raises(ValueError):
        E.load_model(bad)


def test_train_scope_estimate_and_report_label(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    tid = db.insert_training_run(conn, {"model": "m", "strategy": "lora", "dataset": "pubmedqa", "seed": 42,
                                        "learning_rate": 1e-4, "duration_s": 100.0, "peak_mem_mb": 2048.0,
                                        "mem_kind": "RSS-peak", "device": "cpu", "cpu_s": 2400.0, "is_mock": 0})
    db.upsert_cpu_usage(conn, {"scope": "train", "ref_id": str(tid), "device": "cpu", "wall_s": 100.0,
                               "cpu_s": 2400.0, "peak_mem_mb": 2048.0, "mem_kind": "RSS-peak", "source": "container-cgroup"})
    E.compute(conn, E.load_model())
    t = E.summary_tables(conn, load_tariff(DEFAULT_TARIFF))["estimated_energy_training"]
    assert t.energy_j_central.iloc[0] == pytest.approx(2400 * 6.875 + 100 * 2 * 0.3725)
    assert conn.execute("SELECT energy_j FROM training_runs").fetchone()[0] is None


def test_session_records_cpu_seconds():
    class FakeCpu:
        def __init__(self):
            self.v = 10.0

        def cpu_s(self):
            self.v += 1.5
            return self.v

    s = TelemetrySession(cpu=FakeCpu())
    st = s.end(s.begin())
    assert st.cpu_s == pytest.approx(1.5) and st.wall_s >= 0
    assert TelemetrySession().end(TelemetrySession().begin()).cpu_s is None


def test_cli_energy_estimate(tmp_path, capsys):
    from slmbench.cli import main

    conn, csv_path = _pilot_db(tmp_path)
    conn.close()
    rc = main(["energy-estimate", "--db", str(tmp_path / "p.db"), "--cgroup-csv", str(csv_path),
               "--device", "edge-container-4c8g", "--out", str(tmp_path / "o")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["energy_basis"] == "estimated:green-algorithms" and out["estimates"] == 3
    assert "not metered" in (tmp_path / "o" / "estimated_energy.md").read_text()
    assert main(["energy-estimate", "--db", str(tmp_path / "p.db"), "--cgroup-csv", str(csv_path)]) == 2


def test_report_labels_estimates_and_keeps_h1b_unmeasured(tmp_path):
    from slmbench.config import load_matrix
    from slmbench.report import build_report

    conn, csv_path = _pilot_db(tmp_path)
    E.import_host_csv(conn, csv_path, "edge-container-4c8g")
    E.compute(conn, E.load_model())
    man = build_report(conn, load_matrix(ROOT / "configs" / "matrix-pilot.yaml"), load_tariff(DEFAULT_TARIFF),
                       tmp_path / "rep")
    md = (tmp_path / "rep" / "tables" / "estimated_energy_inference.md").read_text()
    assert "estimated from CPU time, not metered" in md and "estimated:green-algorithms" in md
    assert any(f.endswith("estimated_energy_by_precision.png") for f in man["files"])
    dec = (tmp_path / "rep" / "hypothesis_decisions.md").read_text()
    h1b = [line for line in dec.splitlines() if "H1b-energy" in line]
    assert h1b and all("supported" not in line.replace("not supported", "") for line in h1b)
