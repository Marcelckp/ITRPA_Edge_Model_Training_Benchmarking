"""Behavioural changes from review round 1 (.yolo-sisyphus/briefs/prototype-fixes-r1.md)."""

import json
import time

import pandas as pd
import pytest

from slmbench import analysis as A
from slmbench import config as C
from slmbench import db
from slmbench import stats as S
from slmbench import train as T
from slmbench.backends import make_backend
from slmbench.backends.openai_compat import LlamaCppBackend
from slmbench.config import ConfigError
from slmbench.env import collect
from slmbench.export import export, verify_provenance
from slmbench.prompts import build_prompt, render
from slmbench.runner import Runner, RunOptions
from slmbench.scoring import finqa_correct, parse_arc
from slmbench.telemetry import MockPowerSource, TelemetrySession
from slmbench.telemetry.base import EnergyAccumulator
from slmbench.telemetry.usb_meter_csv import NaiveTimestampError, read_meter_csv
from slmbench.template_check import compare


def _opts(data, **kw):
    base = dict(data_dir=data, warmup=0, idle_s=0, repeats=1, cooldown_below_c=None, throttle_reader=None)
    base.update(kw)
    return RunOptions(**base)


def _session():
    return TelemetrySession(power=MockPowerSource(), interval_s=0.01)


def _one(small_matrix, **flt):
    f = {"model": ["toy-a"], "strategy": ["base"], "dataset": ["finqa"], "precision": ["F16"], **flt}
    return C.expand(small_matrix, ["edge"], is_mock=True, filters=f)


def test_fft_recipe_fp32_master_and_fp16_fallback(small_matrix):
    fft = T.make_plan(small_matrix, "toy-a", "fft", "finqa", 42, 1e-5, "data")
    r = T.precision_recipe(fft, bf16_supported=True)
    assert r["load_dtype"] == "float32" and r["bf16"] and r["label"] == "adamw_torch fp32-master bf16-autocast"
    old = T.precision_recipe(fft, bf16_supported=False)
    assert old["fp16"] and not old["bf16"] and "fp16-autocast" in old["label"]
    lora = T.precision_recipe(T.make_plan(small_matrix, "toy-a", "lora", "finqa", 42, 1e-4, "data"), True)
    assert lora["load_dtype"] == "bfloat16"
    q = T.make_plan(small_matrix, "toy-a", "qlora", "finqa", 42, 1e-4, "data")
    assert q["qlora_upcast_non_quantized"] and "fp32 upcast" in T.precision_recipe(q, True)["label"]
    assert "no-upcast" in T.precision_recipe({**q, "qlora_upcast_non_quantized": False}, True)["label"]
    assert fft["logging_steps"] == 10 and fft["chat_template_kwargs"] == {"date_string": "26 Jul 2024"}


def test_meter_idle_uses_same_meter_and_block_interval(tmp_path, small_matrix, prepared_data):
    readings = iter([1000.0, 1001.0, 1011.0])  # idle start, idle end = block start, block end (mWh)
    times = iter([0.0, 60.0, 160.0])
    conn = db.connect(tmp_path / "r.db")
    opts = _opts(prepared_data, idle_s=60, block_energy="usb_meter_manual", meter_prompt=lambda _: next(readings),
                 clock=lambda: next(times), sleep=lambda s: None)
    Runner(conn, make_backend("mock"), _session(), opts).run(_one(small_matrix))
    b = conn.execute("SELECT * FROM energy_blocks").fetchone()
    assert b["source"] == "usb_meter_manual" and b["energy_j"] == pytest.approx(36.0)
    assert b["idle_power_w"] == pytest.approx(3.6 / 60)
    assert b["idle_energy_j"] == pytest.approx(3.6 / 60 * 100)


def test_meter_csv_mode_records_times_only(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    opts = _opts(prepared_data, idle_s=1, block_energy="usb_meter_csv", sleep=lambda s: None)
    Runner(conn, make_backend("mock"), _session(), opts).run(_one(small_matrix))
    b = conn.execute("SELECT * FROM energy_blocks").fetchone()
    assert b["idle_started_at"] and b["started_at"] and b["ended_at"] and b["energy_j"] is None


def test_csv_timestamps_need_time_zone(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("timestamp,mWh\n2026-01-01T02:00:00,1000\n")
    with pytest.raises(NaiveTimestampError):
        read_meter_csv(p)
    sast = read_meter_csv(p, "+02:00")[0]["t"]
    utc = read_meter_csv(p, "UTC")[0]["t"]
    assert utc - sast == pytest.approx(7200)
    assert read_meter_csv(p, "Africa/Johannesburg")[0]["t"] == pytest.approx(sast)


def test_running_trapezoid_is_unbounded():
    acc = EnergyAccumulator(0.0, None)
    for t in range(1, 10001):
        acc.add(float(t), 5.0)
    assert acc.close(10001.0) == pytest.approx(5.0 * 10001)
    s = _session()
    s._samplers["power"].keep_s = 0.02  # far shorter than the window
    s.start()
    try:
        mark = s.begin()
        time.sleep(0.2)
        st = s.end(mark)
    finally:
        s.stop()
    assert st.energy_j == pytest.approx(3.0 * 0.2, rel=0.2)


def test_resumed_block_gets_new_id(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    cfgs = _one(small_matrix)
    Runner(conn, make_backend("mock"), _session(), _opts(prepared_data, limit=3)).run(cfgs)
    Runner(conn, make_backend("mock"), _session(), _opts(prepared_data)).run(cfgs)
    ids = [r[0] for r in conn.execute("SELECT block_id FROM energy_blocks")]
    assert len(ids) == 2 == len(set(ids))
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 6


def test_prompts_built_before_block_queries(tmp_path, small_matrix, prepared_data):
    events = []
    r = Runner(db.connect(tmp_path / "r.db"), make_backend("mock"), _session(), _opts(prepared_data))
    prep, query = r.prepare, r.query
    r.prepare = lambda *a, **k: (events.append("prep"), prep(*a, **k))[1]
    r.query = lambda *a, **k: (events.append("query"), query(*a, **k))[1]
    r.run(_one(small_matrix))
    assert events == ["prep"] * 6 + ["query"] * 6


def test_energy_repeats_interleaved(tmp_path, small_matrix, prepared_data, full_matrix):
    assert full_matrix["runner"]["energy_repeats"] == 3 and RunOptions().repeats == 3
    r = Runner(db.connect(tmp_path / "r.db"), make_backend("mock"), _session(), _opts(prepared_data, repeats=3))
    seen = []
    rb = r.run_block
    r.run_block = lambda cfg, rep, att: (seen.append((rep, cfg.config_id)), rb(cfg, rep, att))[1]
    cfgs = _one(small_matrix, precision=["F16", "Q8_0"])
    r.run(cfgs)
    assert [rep for rep, _ in seen] == [0, 0, 1, 1, 2, 2]
    assert [c for _, c in seen[:2]] == [c for _, c in seen[2:4]] == r.summary.order


def test_scoring_tolerance_and_percent_symmetry():
    assert not finqa_correct("2.6", 3.0).correct
    assert finqa_correct("0.125", "12.5%").correct
    assert finqa_correct("12.5%", 0.125).correct
    assert not finqa_correct("0.125%", "12.5").correct
    assert finqa_correct("1009", 1000.0).correct and not finqa_correct("1011", 1000.0).correct


@pytest.mark.parametrize("text,expected", [("A plant needs light", None), ("a", None), ("B)", "B"),
                                           ("The answer is a plant", None), ("Answer: C", "C"), ("D.", "D")])
def test_arc_parse_ignores_article(text, expected):
    assert parse_arc(text, list("ABCD")) == expected


def test_newcombe_bentur_example_and_zero_discordant():
    a = [1] * 1 + [1] * 1 + [0] * 7 + [0] * 12  # Bentur example: n11=1, n10=1, n01=7, n00=12
    b = [1] * 1 + [0] * 1 + [1] * 7 + [0] * 12
    r = S.newcombe_paired_ci(a, b)
    # Continuity-corrected phi (Newcombe 1998); Fagerland et al. (2014) report -0.507 to -0.026 for these data.
    assert (r["diff"], r["lo"], r["hi"]) == pytest.approx((-0.2857, -0.5069, -0.0256), abs=1e-4)
    for n11, n00 in ((300, 100), (400, 0), (0, 400)):
        z = S.newcombe_paired_ci([1] * n11 + [0] * n00, [1] * n11 + [0] * n00)
        assert z["diff"] == 0 and z["hi"] - z["lo"] > 0 and z["lo"] <= 0 <= z["hi"]


def test_accuracy_figure_has_one_bar_per_precision():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from slmbench.report import _bar_accuracy

    cs = pd.DataFrame([{"plan": "accuracy", "dataset": "d", "model": m, "variant": v, "precision": p,
                        "accuracy": 0.5, "acc_lo": 0.4 + 0.005 * i, "acc_hi": 0.6}
                       for i, (m, v, p) in enumerate((m, v, p) for m in ("a", "b") for v in ("base", "lora")
                                                     for p in ("F16", "Q8_0", "Q4_K_M"))])
    fig, ax = plt.subplots()
    assert _bar_accuracy(cs, "d", "accuracy", ["F16", "Q8_0", "Q4_K_M"])(ax)
    assert len(ax.patches) == 12 and len(ax.get_legend().get_texts()) == 3
    plt.close(fig)


def test_export_provenance(tmp_path, small_matrix):
    conn = db.connect(tmp_path / "r.db")
    for lr, sel in ((1e-4, 0), (2e-4, 1)):
        db.insert_training_run(conn, {"model": "toy-a", "strategy": "lora", "dataset": "finqa", "seed": 42,
                                      "learning_rate": lr, "selected": sel})
    good = C.trained_model_dir(small_matrix, "toy-a", "lora", "finqa", 2e-4, 42)
    tid = verify_provenance(small_matrix, conn, "toy-a", "lora", "finqa", good)
    assert tid == 2
    with pytest.raises(ConfigError):
        verify_provenance(small_matrix, conn, "toy-a", "lora", "finqa",
                          C.trained_model_dir(small_matrix, "toy-a", "lora", "finqa", 1e-4, 42))
    plan = export(small_matrix, conn, "toy-a", "lora", "finqa", good, "/lc", dry_run=True)
    assert plan["train_id"] == 2


def test_config_driven_constants(small_matrix):
    s = A._settings({**small_matrix, "rag_k": [2, 4, 8], "analysis": {"precisions": ["Q8_0", "Q4_K_M"]}})
    assert s["k_levels"] == [2, 4, 8] and s["precisions"] == ["Q8_0", "Q4_K_M"] and s["ft"] == ["fft", "lora", "qlora"]
    item = {"question": "q?", "pre_text": ["word " * 300], "post_text": [], "table": []}
    assert not build_prompt("finqa", item, "full", context_tokens=700, max_new_tokens=32, overhead_tokens=0).truncated
    assert build_prompt("finqa", item, "full", context_tokens=700, max_new_tokens=32, overhead_tokens=400).truncated


def test_adaptation_energy_counts_both_lr_runs_and_pmic_is_not_metered():
    tr = pd.DataFrame([{"model": "m", "strategy": "lora", "dataset": "d", "seed": 42, "energy_j": 10.0},
                       {"model": "m", "strategy": "lora", "dataset": "d", "seed": 42, "energy_j": 30.0},
                       {"model": "m", "strategy": "lora", "dataset": "d", "seed": 43, "energy_j": 99.0}])
    assert A.adaptation_energy(tr, {"selection_seed": 42})[("m", "lora", "d")] == 40.0
    assert not A.metered_source("pmic_partial_lower_bound-trapezoid") and A.metered_source("usb_meter_csv")
    cs = pd.DataFrame([{"config_id": c, "plan": "edge", "model": "m", "strategy": "base", "variant": "base",
                        "precision": "F16", "device": "pi", "dataset": "d", "k": 0, "accuracy": 0.5,
                        "net_energy_j_mean": 2.0, "energy_j_mean": 3.0} for c in ("c1", "c2")])
    blocks = pd.DataFrame([{"config_id": "c1", "source": "pmic_partial_lower_bound-trapezoid", "energy_j": 5.0},
                           {"config_id": "c2", "source": "rapl-package-counter", "energy_j": 5.0}])
    e = A.per_query_energy(cs, pd.DataFrame(), blocks).set_index("config_id")
    assert pd.isna(e.loc["c1", "j_per_query"]) and e.loc["c1", "energy_basis"] == "no metered energy"
    assert e.loc["c2", "j_per_query"] == 2.0


def test_load_frames_filters_mock(tmp_path, small_matrix):
    conn = db.connect(tmp_path / "r.db")
    for cid, mock in (("real", 0), ("sim", 1)):
        db.upsert_config(conn, {"config_id": cid, "model": "toy-a", "strategy": "base", "precision": "F16",
                                "engine": "llamacpp", "tier": "edge", "device": "edge-pi5", "dataset": "finqa",
                                "context_mode": "full", "k": 0, "seed": 42, "temperature": 0.0,
                                "max_new_tokens": 32, "is_mock": mock})
        db.insert_run(conn, {"config_id": cid, "query_id": "q", "correct": 1})
    assert set(A.load_frames(conn, small_matrix)["runs"].config_id) == {"real"}
    assert set(A.load_frames(conn, small_matrix, allow_mock=True)["runs"].config_id) == {"real", "sim"}
    with pytest.raises(A.MockDataRefused):
        A.analyze(conn, small_matrix, C.load_tariff())


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.bodies = []

    def post(self, url, json, timeout):
        self.bodies.append((url, json))
        return _Resp({"prompt": "<s>System: x"})


def test_chat_template_parity_pieces():
    for name in ("gpu-workstation", "edge-pi5", "edge-thinclient", "edge-jetson"):
        assert "--jinja" in C.load_device(name)["backends"]["llamacpp"]["server_cmd"]
    sess = _Session()
    b = LlamaCppBackend({"chat_template_kwargs": {"date_string": "26 Jul 2024"}}, session=sess)
    assert b.apply_template([{"role": "user", "content": "x"}]) == "<s>System: x"
    url, body = sess.bodies[0]
    assert url.endswith("/apply-template") and body["chat_template_kwargs"]["date_string"] == "26 Jul 2024"
    assert b.request_body([], 8, 0.0, 42)["chat_template_kwargs"] == {"date_string": "26 Jul 2024"}
    assert compare("<s>abc", "abc", "<s>")["match"]
    mism = compare("abc\ndef", "abc\nxyz")
    assert not mism["match"] and "-def" in mism["diff"]

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            return json.dumps(kw)

    assert json.loads(render([{"role": "user", "content": "q"}], Tok(), template_kwargs={"date_string": "d"})) == \
        {"date_string": "d"}


def test_env_collects_expected_keys_offline():
    env = collect("devx", llama_cpp_dir="/nonexistent", note="n")
    for key in ("hostname", "os", "python", "cpu", "cpu_count", "ram_mb", "cpu_governor", "slmbench_version",
                "prompt_version", "gpu", "lib/numpy", "lib/torch", "llama_cpp_commit", "note"):
        assert f"devx/{key}" in env, key
    assert env["devx/lib/numpy"] and env["devx/lib/torch"] is None and env["devx/note"] == "n"
