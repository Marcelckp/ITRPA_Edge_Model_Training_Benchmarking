"""`slmbench preflight` with every hardware and network probe faked."""

import pytest

from slmbench import config as C
from slmbench import db
from slmbench import preflight as P
from slmbench.cli import main

NOW = 1_800_000_000.0
RealHost = P.Host


class Src:
    def __init__(self, ok=True, counter=True):
        self.ok, self.counter = ok, counter

    def available(self):
        return self.ok

    @property
    def has_counter(self):
        return self.counter

    def energy_j(self):
        return 1.0 if self.ok else None

    def close(self):
        pass


def host(files=None, cmds=None, http=None, versions=None, exists=None, free_gb=100.0, nvml=None, rapl=None,
         sha=None):
    files = files or {}
    return RealHost(
        read_text=lambda p: files.get(p),
        glob=lambda pat: sorted(k for k in files if _globmatch(pat, k)),
        run=lambda cmd: (cmds or {}).get(cmd[0]),
        http_get=lambda url: (http or {}).get(url, (None, None)),
        disk_free=lambda p: int(free_gb * 1e9),
        version=lambda name: (versions or {}).get(name, "99.0"),
        python=(3, 12),
        nvml=lambda i: nvml or Src(),
        rapl=lambda z: rapl or Src(),
        now=lambda: NOW,
        exists=exists or (lambda p: True),
        sha256=sha or (lambda p: "abc"),
    )


def _globmatch(pat, path):
    import fnmatch

    return fnmatch.fnmatch(path, pat)


def by_name(checks):
    return {c.name: c for c in checks}


PI_FILES = {
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": "performance",
    "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor": "performance",
    "/sys/class/thermal/thermal_zone0/temp": "45000",
    "/sys/class/net/wlan0/operstate": "down",
}


def _meter(tmp_path, ts="2027-01-15T10:39:50+02:00"):
    p = tmp_path / "meter.csv"
    p.write_text(f"timestamp,mWh\n2027-01-15T10:30:00+02:00,100\n{ts},120\n")
    return str(p)


def test_libraries_and_python():
    dev = C.load_device("gpu-workstation")
    ok = by_name(P.check_libraries(dev, host()))
    assert ok["python"].status == ok["libs[gpu]"].status == ok["libs[analysis]"].status == "PASS"
    bad = by_name(P.check_libraries(dev, host(versions={"peft": "0.11.1", "torch": None})))
    assert bad["libs[gpu]"].status == "FAIL" and "peft 0.11.1 < 0.12" in bad["libs[gpu]"].detail
    assert "missing torch" in bad["libs[gpu]"].detail
    assert "libs[edge]" in by_name(P.check_libraries(C.load_device("edge-pi5"), host()))


def test_server_health_and_jinja():
    dev = C.load_device("edge-pi5")
    url = "http://127.0.0.1:8080"
    down = P.check_server(dev, host())
    assert down[0].status == "WARN" and "runner starts it" in down[0].detail
    unmanaged = {**dev, "backends": {"llamacpp": {**dev["backends"]["llamacpp"], "manage_server": False}}}
    assert P.check_server(unmanaged, host())[0].status == "FAIL"
    up = by_name(P.check_server(dev, host(http={f"{url}/health": (200, {"status": "ok"}),
                                                f"{url}/props": (200, {"jinja": True})})))
    assert up["llama-server"].status == "PASS" and up["jinja"].status == "PASS"
    off = by_name(P.check_server(dev, host(http={f"{url}/health": (200, {}), f"{url}/props": (200, {"jinja": False})})))
    assert off["jinja"].status == "FAIL"
    hidden = by_name(P.check_server(dev, host(http={f"{url}/health": (200, {}), f"{url}/props": (200, {})})))
    assert hidden["jinja"].status == "WARN" and "has --jinja" in hidden["jinja"].detail


def test_http_probe_refuses_non_loopback():
    with pytest.raises(ValueError):
        P._http_get("http://10.0.0.5:8080/health")


def test_models_exist_and_sha(tmp_path, full_matrix):
    cfgs = C.expand(full_matrix, ["edge"], only_devices=["edge-pi5"])
    paths = sorted({c.model_path for c in cfgs})
    conn = db.connect(tmp_path / "r.db")
    db.insert_artifact(conn, {"model": "m", "strategy": "s", "dataset": "d", "precision": "F16", "path": paths[0],
                              "size_bytes": 1, "sha256": "abc"})
    ok = by_name(P.check_models(cfgs, conn, host(), True))
    assert ok["model files"].status == "PASS" and ok["model sha256"].status == "WARN"  # others unrecorded
    bad = by_name(P.check_models(cfgs, conn, host(sha=lambda p: "zzz"), True))
    assert bad["model sha256"].status == "FAIL"
    missing = by_name(P.check_models(cfgs, conn, host(exists=lambda p: p != paths[1]), True))
    assert missing["model files"].status == "FAIL" and "1/" in missing["model files"].detail
    assert by_name(P.check_models(cfgs, None, host(), True))["model sha256"].status == "WARN"


def test_governor_pi_rapl_nvml():
    pi = C.load_device("edge-pi5")
    assert P.check_governor(pi, host(PI_FILES)).status == "PASS"
    od = {**PI_FILES, "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor": "ondemand"}
    assert P.check_governor(pi, host(od)).status == "FAIL"
    assert P.check_governor(C.load_device("gpu-workstation"), host()).status == "WARN"
    ok = by_name(P.check_pi(pi, host(PI_FILES, cmds={"vcgencmd": "throttled=0x0"}), 50))
    assert ok["throttled"].status == ok["soc temperature"].status == ok["wifi"].status == "PASS"
    hot = {**PI_FILES, "/sys/class/thermal/thermal_zone0/temp": "61000", "/sys/class/net/wlan0/operstate": "up"}
    bad = by_name(P.check_pi(pi, host(hot, cmds={"vcgencmd": "throttled=0x50005"}), 50))
    assert bad["throttled"].status == bad["soc temperature"].status == "FAIL" and bad["wifi"].status == "WARN"
    thin = C.load_device("edge-thinclient")
    assert P.check_pi(thin, host(), 50)[0].status == "WARN"
    assert P.check_rapl(thin, host()).status == "PASS"
    assert P.check_rapl(thin, host(rapl=Src(ok=False))).status == "FAIL"
    assert P.check_rapl(pi, host()).status == "WARN"
    gpu = C.load_device("gpu-workstation")
    assert P.check_nvml(gpu, host()).status == "PASS"
    assert P.check_nvml(gpu, host(nvml=Src(counter=False))).status == "FAIL"
    assert P.check_nvml(gpu, host(nvml=Src(ok=False))).status == "FAIL"
    assert P.check_nvml(pi, host()).status == "WARN"


def test_meter_csv(tmp_path):
    pi = C.load_device("edge-pi5")
    h = host(exists=lambda p: __import__("os").path.exists(p))
    now_iso = "2027-01-15T10:39:50+02:00"
    t = RealHost(**{**h.__dict__, "now": lambda: __import__("datetime").datetime.fromisoformat(now_iso).timestamp() + 10})
    assert P.check_meter_csv(pi, t, _meter(tmp_path), None, 60).status == "PASS"
    stale = RealHost(**{**t.__dict__, "now": lambda: t.now() + 3600})
    assert "old" in P.check_meter_csv(pi, stale, _meter(tmp_path), None, 60).detail
    naive = _meter(tmp_path, ts="2027-01-15T10:39:50")
    r = P.check_meter_csv({**pi, "telemetry": {**pi["telemetry"], "meter_tz": None}}, t, naive, None, 60)
    assert r.status == "FAIL" and "time zone" in r.detail
    assert P.check_meter_csv(pi, t, str(tmp_path / "none.csv"), None, 60).status == "FAIL"
    assert P.check_meter_csv(C.load_device("edge-thinclient"), t, None, None, 60).status == "WARN"


def test_disk_and_db(tmp_path):
    d = by_name(P.check_disk(str(tmp_path / "r.db"), str(tmp_path / "models"), host(free_gb=1.0), 20, 2))
    assert d["disk[results]"].status == "FAIL"
    assert by_name(P.check_disk(str(tmp_path / "r.db"), str(tmp_path), host(free_gb=10.0), 20, 2))["disk[models]"].status == "WARN"
    absent, conn = P.check_db(str(tmp_path / "new" / "r.db"))
    assert absent[0].status == "PASS" and conn is None and not (tmp_path / "new").exists()
    path = tmp_path / "r.db"
    c = db.connect(path)
    db.insert_training_run(c, {"model": "m", "strategy": "lora", "dataset": "d", "seed": 42, "is_mock": 1})
    c.close()
    checks, conn = P.check_db(str(path))
    conn.close()
    assert by_name(checks)["results db"].status == "PASS" and by_name(checks)["mock rows"].status == "WARN"
    path.chmod(0o444)
    try:
        assert P.check_db(str(path))[0][0].status == "FAIL"
    finally:
        path.chmod(0o644)


def test_run_preflight_and_cli_exit_codes(tmp_path, monkeypatch, capsys):
    pi = C.load_device("edge-pi5")
    good = host(PI_FILES, cmds={"vcgencmd": "throttled=0x0"})
    checks = P.run_preflight(pi, [], str(tmp_path / "r.db"), str(tmp_path), good, meter_csv=None)
    statuses = by_name(checks)
    assert statuses["meter csv"].status == "FAIL"  # configured path is not a real file here
    table = P.format_table(checks)
    assert "STATUS" in table and "FAIL" in table
    monkeypatch.setattr(P, "Host", lambda: host(PI_FILES, cmds={"vcgencmd": "throttled=0x5"}))
    rc = main(["preflight", "--device", "edge-pi5", "--db", str(tmp_path / "x.db"), "--no-sha"])
    out = capsys.readouterr().out
    assert rc == 1 and "throttled" in out and "FAIL" in out
    assert not (tmp_path / "x.db").exists()  # read-only: the DB is not created
    monkeypatch.setattr(P, "Host", lambda: host())
    assert main(["preflight", "--device", "gpu-workstation", "--db", str(tmp_path / "x.db")]) == 0
    assert "0 FAIL" in capsys.readouterr().out
