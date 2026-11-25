from slmbench import config as C
from slmbench import db
from slmbench.backends import make_backend
from slmbench.runner import Runner, RunOptions, block_order
from slmbench.telemetry import MockPowerSource, TelemetrySession


def _opts(data, **kw):
    base = dict(data_dir=data, warmup=2, idle_s=0.02, repeats=1, cooldown_below_c=None, throttle_reader=None,
                subset_strategies={"pubmedqa": "stratified"})
    base.update(kw)
    return RunOptions(**base)


def _session():
    return TelemetrySession(power=MockPowerSource(), temperature=MockPowerSource(), interval_s=0.01)


def _edge(small_matrix, **flt):
    return C.expand(small_matrix, ["edge"], is_mock=True, filters=flt or None)


def test_block_order_seeded_and_input_independent(small_matrix):
    cfgs = _edge(small_matrix)
    a = [c.config_id for c in block_order(cfgs, 42)]
    b = [c.config_id for c in block_order(list(reversed(cfgs)), 42)]
    assert a == b and a != sorted(a)
    assert a != [c.config_id for c in block_order(cfgs, 7)]


def test_run_resumable_with_warmup_and_blocks(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    cfgs = _edge(small_matrix, model=["toy-a"], strategy=["base", "rag"], dataset=["pubmedqa"], precision=["Q4_K_M"])
    s1 = Runner(conn, make_backend("mock"), _session(), _opts(prepared_data)).run(cfgs)
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == len(cfgs) * 6 == s1.inserted
    assert s1.warmups >= 2
    row = conn.execute("SELECT * FROM runs WHERE retrieval_hit IS NOT NULL LIMIT 1").fetchone()
    assert row["is_mock"] == 1 and "-r0-a0-" in row["block_id"] and row["parse_ok"] in (0, 1)
    assert row["server_prompt_ms"] is not None and row["energy_j"] > 0
    blk = conn.execute("SELECT * FROM energy_blocks").fetchall()
    assert len(blk) == len(cfgs) and all(b["n_queries"] == 6 and b["idle_power_w"] for b in blk)
    s2 = Runner(conn, make_backend("mock"), _session(), _opts(prepared_data)).run(cfgs)
    assert s2.inserted == 0 and s2.skipped_blocks == len(cfgs)


def test_subset_identical_and_stratified(small_matrix, prepared_data, tmp_path):
    cfgs = _edge(small_matrix, dataset=["pubmedqa"])
    r = Runner(db.connect(tmp_path / "r.db"), make_backend("mock"), _session(), _opts(prepared_data))
    sets = {tuple(i["query_id"] for i in r.items_for(c)) for c in cfgs}
    assert len(sets) == 1
    golds = [i["gold"] for i in r.items_for(cfgs[0])]
    assert sorted(golds) == ["maybe", "maybe", "no", "no", "yes", "yes"]


def test_energy_repeat_blocks(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    cfgs = _edge(small_matrix, model=["toy-a"], strategy=["base"], dataset=["finqa"])
    Runner(conn, make_backend("mock"), _session(), _opts(prepared_data, repeats=3)).run(cfgs)
    assert conn.execute("SELECT COUNT(*) FROM energy_blocks").fetchone()[0] == 3 * len(cfgs)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 6 * len(cfgs)


def test_throttled_block_is_rerun(tmp_path, small_matrix, prepared_data):
    conn = db.connect(tmp_path / "r.db")
    cfgs = _edge(small_matrix, model=["toy-a"], strategy=["base"], dataset=["finqa"], precision=["F16"])
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        return 0x4 if calls["n"] <= 3 else 0x0

    s = Runner(conn, make_backend("mock"), _session(), _opts(prepared_data, throttle_reader=reader)).run(cfgs)
    assert s.throttled_reruns == 1
    rows = conn.execute("SELECT block_id, throttled FROM runs").fetchall()
    assert len(rows) == 6 and {r["block_id"].rsplit("-", 1)[0] for r in rows} == {f"{cfgs[0].config_id}-r0-a1"}


def test_cooldown_gate_waits(tmp_path, small_matrix, prepared_data):
    temps = iter([60.0, 55.0, 45.0] + [40.0] * 100)

    class Hot(MockPowerSource):
        def read(self):
            r = super().read()
            r.temp_c = next(temps)
            return r

    sess = TelemetrySession(power=MockPowerSource(), temperature=Hot(), interval_s=1.0)
    slept = []
    opts = _opts(prepared_data, cooldown_below_c=50.0, cooldown_poll_s=1.0, sleep=slept.append, idle_s=0)
    cfgs = _edge(small_matrix, model=["toy-a"], strategy=["base"], dataset=["finqa"], precision=["F16"])
    r = Runner(db.connect(tmp_path / "r.db"), make_backend("mock"), sess, opts)
    r._cooldown()
    assert slept == [1.0, 1.0]
    assert cfgs


def test_backend_errors_recorded_not_raised(tmp_path, small_matrix, prepared_data):
    class Broken:
        is_mock = True
        server_pid = None

        def load(self, cfg):
            pass

        def unload(self):
            pass

        def count_tokens(self, text):
            return None

        def generate(self, *a, **k):
            raise RuntimeError("server gone")

    conn = db.connect(tmp_path / "r.db")
    cfgs = _edge(small_matrix, model=["toy-a"], strategy=["base"], dataset=["finqa"], precision=["F16"])
    s = Runner(conn, Broken(), _session(), _opts(prepared_data)).run(cfgs)
    assert s.errors == 6
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE error LIKE 'RuntimeError%'").fetchone()[0] == 6
