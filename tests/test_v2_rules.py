"""DESIGN.md v2 decision rules and measurement controls not covered elsewhere."""

import sqlite3
import time

import numpy as np
import pandas as pd

from slmbench import analysis as A
from slmbench import config as C
from slmbench import db
from slmbench.backends import make_backend
from slmbench.cli import main
from slmbench.runner import Runner, RunOptions
from slmbench.telemetry import MockPowerSource, TelemetrySession
from slmbench.telemetry.pi_pmic import PiPmicSource


def _s():
    return A._settings({"analysis": {"bootstrap_resamples": 500}})


def test_h1a_ordering_rule():
    rows = []
    for model, vram in (("a", {"fft": 3000, "lora": 2000, "qlora": 1000}),
                        ("b", {"fft": 3000, "lora": 900, "qlora": 1000})):
        for strat, v in vram.items():
            for seed in (42, 43, 44):
                rows.append({"dataset": "pubmedqa", "model": model, "strategy": strat, "seed": seed, "selected": int(seed == 42),
                             "peak_vram_mb": v + seed, "duration_s": 1.0, "energy_j": 1.0, "trainable_params": 1,
                             "total_params": 1, "is_mock": 0})
    dec, summ = A.h1a(pd.DataFrame(rows), _s())
    assert dec[0]["decision"] == "not supported" and "b:fails" in dec[0]["effect"]
    assert summ.peak_vram_cv.notna().all()


def test_h1b_block_energy_rule():
    def blocks(vals, n):
        return [{"tier": "edge", "device": "pi", "dataset": "d", "model": "m", "variant": "base", "precision": p,
                 "energy_j": 1.0, "net_j_per_query": v + 0.01 * i, "gross_j_per_query": v, "source": "usb_meter_manual",
                 "is_mock": 0} for p, v in vals.items() for i in range(n)]
    sep = A.block_energy(pd.DataFrame(blocks({"F16": 3.0, "Q8_0": 2.0, "Q4_K_M": 1.0}, 3)), _s())[0]
    assert sep[0]["decision"] == "supported"
    over = A.block_energy(pd.DataFrame(blocks({"F16": 1.0, "Q8_0": 1.0, "Q4_K_M": 1.0}, 3)), _s())[0]
    assert over[0]["decision"] == "not supported"
    few = A.block_energy(pd.DataFrame(blocks({"F16": 3.0, "Q8_0": 2.0, "Q4_K_M": 1.0}, 2)), _s())[0]
    assert few[0]["decision"] == "not testable"


def test_h2b_one_sided_mcnemar_with_holm():
    rng = np.random.default_rng(3)
    rows = []
    for strat, p in (("base", 0.6), ("fft", 0.3), ("lora", 0.7), ("qlora", 0.3)):
        vals = (rng.random(200) < p).astype(int)
        for i, v in enumerate(vals):
            rows.append({"plan": "retention", "dataset": "arc_easy", "train_domain": "" if strat == "base" else "pubmedqa",
                         "device": "gpu", "model": "m", "precision": "F16", "strategy": strat, "query_id": f"q{i}",
                         "correct": v, "error": None})
    res = A.h2b(pd.DataFrame(rows), _s(), ["pubmedqa"])
    A._apply_holm(res, 0.05)
    lora = next(r for r in res if r["comparison"].startswith("lora"))
    assert lora["test"] == "mcnemar-exact" and "one-sided" in lora["comparison"]
    assert lora["decision"] == "supported" and lora["p_holm"] < 0.05
    qlora = next(r for r in res if r["comparison"].startswith("qlora"))
    assert qlora["decision"] in ("not supported", "not testable")


def test_h3_page_l_and_diminishing_returns():
    rows = []
    for i in range(20):
        for k in (3, 5, 10):
            rows.append({"strategy": "rag", "tier": "edge", "plan": "edge", "device": "pi", "dataset": "d", "model": "m",
                         "precision": "Q4_K_M", "k": k, "query_id": f"q{i}", "error": None, "correct": 1,
                         "ttft_primary_s": k * 0.1 + i * 0.001, "e2e_latency_s": k * 0.2 + i * 0.002})
    k10 = set(range(8, 20)) | set(range(30, 100))  # 12 gained vs K5, 10 lost: 22 discordant pairs
    acc = {3: [int(i >= 50) for i in range(100)], 5: [int(i >= 20) for i in range(100)],
           10: [int(i in k10) for i in range(100)]}
    for k, vals in acc.items():
        for i, v in enumerate(vals):
            rows.append({"strategy": "rag", "tier": "gpu", "plan": "accuracy", "device": "gpu", "dataset": "d", "model": "m",
                         "precision": "Q4_K_M", "k": k, "query_id": f"a{i}", "error": None, "correct": v,
                         "ttft_primary_s": 0.1, "e2e_latency_s": 0.1})
    res = A.h3(pd.DataFrame(rows), _s())
    A._apply_holm(res, 0.05)
    page = [r for r in res if r["test"] == "page-l"]
    assert len(page) == 2 and all(r["p_holm"] < 0.05 and r["decision"] == "supported" for r in page)
    dim = next(r for r in res if r["comparison"] == "diminishing accuracy returns")
    assert dim["reason"] is None and dim["p_holm"] >= 0.05 and dim["decision"] == "supported"


def test_h3_floor_not_overridden_by_ci():
    rows = []
    acc = {3: [0] * 50 + [1] * 50, 5: [0] * 20 + [1] * 80, 10: [0] * 18 + [1] * 82}  # 2 discordant pairs
    for k, vals in acc.items():
        for i, v in enumerate(vals):
            for plan, tier, dev in (("accuracy", "gpu", "gpu"), ("edge", "edge", "pi")):
                rows.append({"strategy": "rag", "tier": tier, "plan": plan, "device": dev, "dataset": "d", "model": "m",
                             "precision": "F16", "k": k, "query_id": f"q{i}", "error": None, "correct": v,
                             "ttft_primary_s": k * 0.1 + i * 1e-3, "e2e_latency_s": k * 0.2 + i * 1e-3})
    res = A.h3(pd.DataFrame(rows), _s())
    A._apply_holm(res, 0.05)
    dim = next(r for r in res if r["comparison"] == "diminishing accuracy returns")
    assert dim["decision"] == "not testable"


def test_h1b_memory_requires_predicted_direction():
    rows = []
    for i in range(30):
        for p, mem in (("F16", 800), ("Q8_0", 1300), ("Q4_K_M", 2400)):  # reversed ordering
            rows.append({"tier": "edge", "device": "pi", "dataset": "d", "model": "m", "variant": "base", "precision": p,
                         "query_id": f"q{i}", "error": None, "peak_mem_mb": mem + i})
    res, _ = A.h1b_memory(pd.DataFrame(rows), _s())
    A._apply_holm(res, 0.05)
    assert res[0]["p_holm"] < 0.05 and res[0]["decision"] == "not supported"


def test_accuracy_parity_98_percent_flag():
    def rows(plan, tier, device, cid, answers):
        return [{"plan": plan, "tier": tier, "device": device, "config_id": cid, "error": None, "model": "m",
                 "strategy": "base", "precision": "F16", "dataset": "d", "context_mode": "closed", "k": 0,
                 "query_id": f"q{i}", "parsed_answer": a} for i, a in enumerate(answers)]
    ref = ["yes"] * 100
    ok = rows("edge", "edge", "pi", "e1", ["yes"] * 99 + ["no"])
    bad = rows("edge", "edge", "thin", "e2", ["yes"] * 97 + ["no"] * 3)
    par = A.accuracy_parity(pd.DataFrame(rows("accuracy", "gpu", "gpu", "a", ref) + ok + bad), _s()).set_index("device")
    assert par.loc["pi", "agreement"] == 0.99 and not par.loc["pi", "flag_rerun_500"]
    assert par.loc["thin", "flag_rerun_500"]


def test_three_warmups_discarded_per_server_start(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    cfgs = C.expand(small_matrix, ["edge"], is_mock=True,
                    filters={"model": ["toy-a"], "strategy": ["base", "rag"], "dataset": ["pubmedqa"], "precision": ["F16"]})
    assert len({c.model_path for c in cfgs}) == 1
    opts = RunOptions(data_dir=prepared_data, warmup=small_matrix["runner"]["warmup_queries"] + 2, idle_s=0,
                      cooldown_below_c=None, throttle_reader=None)
    assert opts.warmup == 3 and RunOptions().warmup == 3
    s = Runner(conn, make_backend("mock"), TelemetrySession(power=MockPowerSource(), interval_s=0.01), opts).run(cfgs)
    assert s.warmups == 3
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 6 * len(cfgs) == s.inserted


def test_default_runner_controls(full_matrix):
    r = full_matrix["runner"]
    assert (r["warmup_queries"], r["idle_baseline_s"], r["block_order_seed"], r["cooldown_below_c"]) == (3, 60, 42, 50)
    assert full_matrix["training"]["lr_grid"] == {"fft": [1e-5, 2e-5], "lora": [1e-4, 2e-4], "qlora": [1e-4, 2e-4],
                                                  "celora": [1e-4, 2e-4]}
    assert full_matrix["analysis"]["noninferiority_delta"] == 0.05 and full_matrix["analysis"]["parity_threshold"] == 0.98


def test_telemetry_facts():
    assert PiPmicSource.name == "pmic_partial_lower_bound" and PiPmicSource.min_interval_s >= 0.5
    gpu = C.load_device("gpu-workstation")
    assert gpu["telemetry"]["interval_s"] >= 0.1 and gpu["telemetry"]["power"] == ["nvml"]
    pi = C.load_device("edge-pi5")
    assert pi["telemetry"]["block_energy"] == "usb_meter_csv"
    for name in ("gpu-workstation", "edge-pi5", "edge-thinclient", "edge-jetson", "edge-container-4c8g",
                 "cpu-reference"):
        cmd = C.load_device(name)["backends"]["llamacpp"]["server_cmd"]
        # llama.cpp b12532 rejects --mlock; every tier must run the same engine flags.
        assert "--mlock" not in cmd
        assert cmd[cmd.index("--load-mode") + 1] == "mmap+mlock" and cmd[cmd.index("--cache-ram") + 1] == "0"
        assert cmd[cmd.index("--fit") + 1] == "off" and "--jinja" in cmd


def test_energy_block_import_csv(tmp_path):
    dbp = tmp_path / "e.db"
    assert main(["energy-block", "--db", str(dbp), "start", "--block-id", "b1", "--config-id", "c", "--device", "edge-pi5",
                 "--meter-mwh", "0"]) == 0
    time.sleep(0.01)
    assert main(["energy-block", "--db", str(dbp), "stop", "--block-id", "b1", "--meter-mwh", "0", "--n-queries", "10"]) == 0
    now = time.time()
    csv = tmp_path / "m.csv"
    csv.write_text(f"timestamp,mWh\n{now - 100},1000\n{now + 100},1200\n")
    assert main(["energy-block", "--db", str(dbp), "import-csv", "--device", "edge-pi5", "--csv", str(csv)]) == 0
    src, e0, e1, ej = sqlite3.connect(dbp).execute(
        "SELECT source, meter_start_mwh, meter_end_mwh, energy_j FROM energy_blocks").fetchone()
    assert src == "usb_meter_csv" and 1000 < e0 <= e1 < 1200 and ej == (e1 - e0) * 3.6
