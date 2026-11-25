"""Review round 2 fixes."""

import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from slmbench import analysis as A
from slmbench import config as C
from slmbench import db
from slmbench import stats as S
from slmbench import train as T
from slmbench.backends import make_backend
from slmbench.cli import main
from slmbench.prompts import build_prompt
from slmbench.runner import Runner, RunOptions
from slmbench.scoring import finqa_correct
from slmbench.telemetry import MockPowerSource, TelemetrySession


def _block(conn, bid, device, source, t0, t1, idle_t0=None, idle_w=None):
    db.upsert_energy_block(conn, {"block_id": bid, "config_id": "c", "device": device, "source": source,
                                  "started_at": t0, "ended_at": t1, "idle_started_at": idle_t0,
                                  "idle_power_w": idle_w, "energy_j": 99.0, "n_queries": 10})


def test_import_csv_only_touches_meter_blocks_of_device(tmp_path):
    dbp = tmp_path / "e.db"
    conn = db.connect(dbp)
    iso = lambda s: f"2026-01-01T00:{s // 60:02d}:{s % 60:02d}+00:00"  # noqa: E731
    _block(conn, "meter", "edge-pi5", "usb_meter_csv", iso(60), iso(160), iso(0), idle_w=7.0)
    _block(conn, "pmic", "edge-pi5", "pmic_partial_lower_bound-trapezoid", iso(60), iso(160), idle_w=1.0)
    _block(conn, "other", "edge-thinclient", "usb_meter_csv", iso(60), iso(160), iso(0))
    conn.close()
    csv = tmp_path / "m.csv"
    csv.write_text("timestamp,mWh\n" + "".join(f"{iso(s)},{1000 + s / 60:.6f}\n" for s in range(0, 200, 20)))
    base = ["energy-block", "--db", str(dbp), "import-csv", "--csv", str(csv)]
    assert main(base) == 2
    assert main([*base, "--device", "edge-pi5"]) == 0
    rows = {r[0]: r for r in sqlite3.connect(dbp).execute(
            "SELECT block_id, source, energy_j, idle_power_w, idle_energy_j FROM energy_blocks")}
    assert rows["meter"][2] == pytest.approx(100 / 60 * 3.6)
    assert rows["meter"][3] == pytest.approx(3.6 / 60)  # idle recomputed from the CSV, not the stored 7 W
    assert rows["meter"][4] == pytest.approx(rows["meter"][3] * 100)
    assert rows["pmic"][1:4] == ("pmic_partial_lower_bound-trapezoid", 99.0, 1.0)
    assert rows["other"][2] == 99.0


def test_newcombe_corrected_phi_zero_discordant():
    r = S.newcombe_paired_ci([1] * 250 + [0] * 250, [1] * 250 + [0] * 250)
    assert r["diff"] == 0 and r["hi"] - r["lo"] > 0.005 and r["lo"] < 0 < r["hi"]
    assert S.noninferiority([1] * 250 + [0] * 250, [1] * 250 + [0] * 250)["noninferior"] is True


def _fake_torch(available=True, emul=None, cap=(7, 5)):
    def is_bf16_supported(including_emulation=True):
        if emul is None:
            raise TypeError("old torch")
        return emul
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, is_bf16_supported=is_bf16_supported,
                                                get_device_capability=lambda: cap))


def test_native_bf16_and_qlora_compute_dtype(small_matrix):
    assert not T.native_bf16(_fake_torch(emul=False))
    assert T.native_bf16(_fake_torch(emul=True))
    assert not T.native_bf16(_fake_torch(cap=(7, 5))) and T.native_bf16(_fake_torch(cap=(8, 6)))
    assert not T.native_bf16(_fake_torch(available=False))
    q = T.make_plan(small_matrix, "toy-a", "qlora", "finqa", 42, 1e-4, "data")
    assert T.precision_recipe(q, False)["bnb_compute_dtype"] == "float16"
    assert T.precision_recipe(q, True)["bnb_compute_dtype"] == "bfloat16"


def test_gpu_extras_pinned():
    text = (C.PACKAGE_ROOT / "pyproject.toml").read_text()
    for pin in ("peft>=0.12", "transformers>=", "bitsandbytes>=", "accelerate>="):
        assert pin in text


def test_training_prompt_uses_configured_overhead(small_matrix):
    item = {"question": "q?", "pre_text": ["word " * 300], "post_text": [], "table": [], "gold": "1"}
    m = {**small_matrix, "context_tokens": 700, "prompt_overhead_tokens": 400}
    plan = T.make_plan(m, "toy-a", "lora", "finqa", 42, 1e-4, "data")
    assert plan["prompt_overhead_tokens"] == 400
    train_p = T.training_prompt("finqa", item, plan)
    eval_p = build_prompt("finqa", item, "full", None, 700, plan["max_new_tokens"], overhead_tokens=400)
    assert train_p.truncated and train_p.messages == eval_p.messages
    assert not T.training_prompt("finqa", item, {**plan, "prompt_overhead_tokens": 0}).truncated


def test_partial_blocks_excluded_from_h1b():
    rows = []
    for p, v in {"F16": 3.0, "Q8_0": 2.0, "Q4_K_M": 1.0}.items():
        for i in range(3):
            rows.append({"tier": "edge", "device": "pi", "dataset": "d", "model": "m", "variant": "base", "precision": p,
                         "energy_j": 1.0, "net_j_per_query": v + 0.01 * i, "gross_j_per_query": v,
                         "source": "usb_meter_csv", "is_mock": 0, "n_queries": 100, "planned_n": 100})
    partial = {**rows[0], "net_j_per_query": 50.0, "n_queries": 40}
    s = A._settings({"analysis": {"bootstrap_resamples": 500}})
    dec, summ = A.block_energy(pd.DataFrame(rows + [partial]), s)
    assert dec[0]["decision"] == "supported" and dec[0]["partial_blocks"] == 1
    f16 = summ[summ.precision == "F16"].iloc[0]
    assert f16.n_blocks == 3 and f16.partial_blocks == 1 and f16.net_j_per_query < 4
    assert A.planned_items({"plan": "edge", "dataset": "finqa"}, C.load_matrix()) == 100


def test_manual_meter_rejects_non_monotonic(tmp_path, small_matrix, prepared_data):
    answers = iter(["1000", "990", "oops", "1001", "1000.5", "1011"])
    prompts = []

    def prompt(msg):
        prompts.append(msg)
        return next(answers)

    cfgs = C.expand(small_matrix, ["edge"], is_mock=True,
                    filters={"model": ["toy-a"], "strategy": ["base"], "dataset": ["finqa"], "precision": ["F16"]})
    conn = db.connect(tmp_path / "r.db")
    opts = RunOptions(data_dir=prepared_data, warmup=0, idle_s=60, repeats=1, cooldown_below_c=None,
                      throttle_reader=None, block_energy="usb_meter_manual", meter_prompt=prompt, sleep=lambda s: None)
    Runner(conn, make_backend("mock"), TelemetrySession(power=MockPowerSource(), interval_s=0.01), opts).run(cfgs)
    b = conn.execute("SELECT idle_meter_start_mwh, meter_start_mwh, meter_end_mwh FROM energy_blocks").fetchone()
    assert tuple(b) == (1000.0, 1001.0, 1011.0)
    assert len(prompts) == 6 and "must be >=" in prompts[2] and "not a number" in prompts[3] and "must be >=" in prompts[5]
    assert C.load_device("edge-pi5")["telemetry"]["block_energy"] == "usb_meter_csv"


def test_percent_scaled_only_when_one_side_has_percent():
    assert not finqa_correct("1250%", "12.5%").correct
    assert finqa_correct("12.5%", "12.5%").correct
    assert finqa_correct("12.5%", 0.125).correct and finqa_correct("0.125", "12.5%").correct
    assert finqa_correct("12.5", 0.125, gold_text="12.5%").correct
    assert not finqa_correct("1250%", 0.125, gold_text="12.5%").correct
