"""Read-only readiness check for a measurement host. Every probe goes through ``Host`` so tests can fake it."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .config import PACKAGE_ROOT, ExpConfig

log = logging.getLogger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
TIER_EXTRAS = {"gpu": ("gpu", "rag", "analysis"), "edge": ("edge", "rag", "analysis"),
               "cpu": ("edge", "rag", "analysis")}
_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:>=\s*([0-9][0-9A-Za-z.]*))?")


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def _http_get(url: str, timeout: float = 5.0) -> tuple[int | None, Any]:
    """GET a loopback URL; returns (status, parsed JSON or text). Non-loopback hosts are refused."""
    if (urlparse(url).hostname or "") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"preflight only probes loopback endpoints, got {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError):
        return None, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def _version(pkg: str) -> str | None:
    for name in (pkg, pkg.replace("_", "-"), pkg.replace("-", "_")):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _nvml(index: int) -> Any:
    from .telemetry.nvml import NvmlSource

    return NvmlSource(index)


def _rapl(zone: str) -> Any:
    from .telemetry.rapl import RaplSource

    return RaplSource(zone)


@dataclass
class Host:
    read_text: Callable[[str], str | None] = _read_text
    glob: Callable[[str], list[str]] = field(default=lambda p: sorted(glob.glob(p)))
    run: Callable[[list[str]], str | None] = _run
    http_get: Callable[[str], tuple[int | None, Any]] = _http_get
    disk_free: Callable[[str], int] = field(default=lambda p: shutil.disk_usage(p).free)
    version: Callable[[str], str | None] = _version
    python: tuple[int, int] = field(default_factory=lambda: tuple(sys.version_info[:2]))
    nvml: Callable[[int], Any] = _nvml
    rapl: Callable[[str], Any] = _rapl
    now: Callable[[], float] = time.time
    exists: Callable[[str], bool] = os.path.exists
    sha256: Callable[[str], str] | None = None


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def _extras(pyproject: Path) -> tuple[str, dict[str, list[str]]]:
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text())
        proj = data["project"]
        return proj.get("requires-python", ">=3.10"), proj.get("optional-dependencies", {})
    except (ImportError, OSError, KeyError):
        return ">=3.10", {}


def check_libraries(dev: dict, host: Host, pyproject: Path = PACKAGE_ROOT / "pyproject.toml") -> list[Check]:
    req_py, extras = _extras(pyproject)
    want = _vtuple(req_py)
    out = [Check("python", PASS if host.python >= want[:2] else FAIL,
                 f"{host.python[0]}.{host.python[1]} (requires {req_py})")]
    tier = dev.get("tier", "edge")
    for extra in TIER_EXTRAS.get(tier, ()):
        missing, old = [], []
        for req in extras.get(extra, []):
            m = _REQ_RE.match(req)
            if not m:
                continue
            name, minimum = m.group(1), m.group(2)
            have = host.version(name)
            if have is None:
                missing.append(name)
            elif minimum and _vtuple(have) < _vtuple(minimum):
                old.append(f"{name} {have} < {minimum}")
        problems = [f"missing {', '.join(missing)}"] * bool(missing) + old
        out.append(Check(f"libs[{extra}]", FAIL if problems else PASS,
                         "; ".join(problems) or f"{len(extras.get(extra, []))} packages OK"))
    return out


def check_server(dev: dict, host: Host) -> list[Check]:
    b = dev.get("backends", {}).get("llamacpp")
    if not b:
        return [Check("llama-server", WARN, "no llamacpp backend in device profile")]
    url = str(b.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
    status, _ = host.http_get(f"{url}/health")
    if status != 200:
        if b.get("manage_server"):
            return [Check("llama-server", WARN, f"{url} not reachable; the runner starts it (manage_server: true)")]
        return [Check("llama-server", FAIL, f"{url}/health not reachable and manage_server is false")]
    out = [Check("llama-server", PASS, f"{url}/health 200")]
    pstatus, props = host.http_get(f"{url}/props")
    jinja = None
    if pstatus == 200 and isinstance(props, dict):
        for key in ("jinja", "use_jinja"):
            if key in props:
                jinja = bool(props[key])
    if jinja is None:
        cmd_has = "--jinja" in (b.get("server_cmd") or [])
        out.append(Check("jinja", WARN, "server does not expose its jinja setting; profile server_cmd "
                         + ("has --jinja" if cmd_has else "LACKS --jinja")))
    else:
        out.append(Check("jinja", PASS if jinja else FAIL, f"/props reports jinja={jinja}"))
    return out


def check_models(configs: list[ExpConfig], conn: sqlite3.Connection | None, host: Host, verify_sha: bool) -> list[Check]:
    paths = sorted({c.model_path for c in configs if c.model_path.endswith(".gguf")})
    if not paths:
        return [Check("model files", WARN, "plan references no GGUF files (HF ids are resolved at load)")]
    missing = [p for p in paths if not host.exists(p)]
    out = [Check("model files", FAIL if missing else PASS,
                 f"{len(missing)}/{len(paths)} missing, e.g. {missing[0]}" if missing else f"{len(paths)} present")]
    recorded: dict[str, str] = {}
    if conn is not None and _has_table(conn, "artifacts"):
        recorded = {r[0]: r[1] for r in conn.execute("SELECT path, sha256 FROM artifacts WHERE sha256 IS NOT NULL")}
    present = [p for p in paths if p not in missing]
    if not recorded:
        out.append(Check("model sha256", WARN, "no artifacts recorded in the DB (copy the workstation DB to compare)"))
    elif not verify_sha:
        out.append(Check("model sha256", WARN, "skipped (--no-sha)"))
    else:
        from .export import sha256_file

        sha = host.sha256 or sha256_file
        bad = [p for p in present if p in recorded and sha(p) != recorded[p]]
        unrec = [p for p in present if p not in recorded]
        status = FAIL if bad else (WARN if unrec else PASS)
        detail = f"{len(bad)} mismatched" + (f" (e.g. {bad[0]})" if bad else "") + f"; {len(unrec)} not recorded"
        out.append(Check("model sha256", status, detail))
    return out


def _is_pi(dev: dict) -> bool:
    return "pi_pmic" in dev.get("telemetry", {}).get("power", [])


def check_governor(dev: dict, host: Host) -> Check:
    if dev.get("tier") != "edge":
        return Check("cpu governor", WARN, "not applicable (non-edge device)")
    files = host.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")
    govs = {host.read_text(f) for f in files} - {None}
    if not govs:
        return Check("cpu governor", WARN, "cpufreq not exposed")
    ok = govs == {"performance"}
    return Check("cpu governor", PASS if ok else FAIL, f"{','.join(sorted(govs))} on {len(files)} CPUs"
                 + ("" if ok else "; run scripts/pin_cpu_governor.sh performance"))


def check_pi(dev: dict, host: Host, max_temp_c: float) -> list[Check]:
    if not _is_pi(dev):
        return [Check("pi throttle/temp", WARN, "not applicable (no Pi PMIC in profile)"),
                Check("wifi", WARN, "not applicable")]
    out = []
    raw = host.run(["vcgencmd", "get_throttled"])
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", raw or "")
    if not m:
        out.append(Check("throttled", FAIL, "vcgencmd get_throttled unavailable"))
    else:
        v = int(m.group(1), 16)
        out.append(Check("throttled", PASS if v == 0 else FAIL, f"{m.group(1)}" + ("" if v == 0 else
                         " (non-zero: under-voltage/throttling since boot; check PSU and cooler, reboot)")))
    t = host.read_text("/sys/class/thermal/thermal_zone0/temp")
    if t is None:
        out.append(Check("soc temperature", FAIL, "thermal_zone0 unreadable"))
    else:
        c = int(t) / 1000.0
        out.append(Check("soc temperature", PASS if c < max_temp_c else FAIL, f"{c:.1f} C (limit {max_temp_c:g} C)"))
    up = [f.split("/")[-2] for f in host.glob("/sys/class/net/wlan*/operstate") if host.read_text(f) == "up"]
    out.append(Check("wifi", WARN if up else PASS, f"{','.join(up)} up (sudo rfkill block wifi)" if up else "down"))
    return out


def check_rapl(dev: dict, host: Host) -> Check:
    tel = dev.get("telemetry", {})
    if "rapl" not in tel.get("power", []):
        return Check("rapl", WARN, "not applicable (RAPL not in profile)")
    src = host.rapl(tel.get("rapl_zone", "intel-rapl:0"))
    if src.available() and src.energy_j() is not None:
        return Check("rapl", PASS, "energy_uj readable")
    return Check("rapl", FAIL, "energy_uj unreadable (sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj)")


def check_nvml(dev: dict, host: Host) -> Check:
    tel = dev.get("telemetry", {})
    if "nvml" not in tel.get("power", []):
        return Check("nvml", WARN, "not applicable (NVML not in profile)")
    src = host.nvml(int(tel.get("gpu_index", 0)))
    try:
        if not src.available():
            return Check("nvml", FAIL, "no NVML device (driver or nvidia-ml-py missing)")
        if not src.has_counter:
            return Check("nvml", FAIL, "total-energy counter unsupported; per-query energy would need sampling")
        return Check("nvml", PASS, "device present, energy counter supported")
    finally:
        src.close()


def check_meter_csv(dev: dict, host: Host, csv_path: str | None, tz: str | None, max_age_s: float) -> Check:
    tel = dev.get("telemetry", {})
    if tel.get("block_energy") != "usb_meter_csv":
        return Check("meter csv", WARN, "not applicable (block energy source is not usb_meter_csv)")
    path = csv_path or tel.get("meter_csv")
    if not path:
        return Check("meter csv", FAIL, "no CSV path (set telemetry.meter_csv or pass --meter-csv)")
    path = os.path.expanduser(path)
    if not host.exists(path):
        return Check("meter csv", FAIL, f"{path} not found")
    from .telemetry.usb_meter_csv import read_meter_csv

    try:
        rows = read_meter_csv(path, tz or tel.get("meter_tz"))
    except (ValueError, OSError) as exc:
        return Check("meter csv", FAIL, str(exc))
    if not rows:
        return Check("meter csv", FAIL, f"{path} has no usable rows")
    age = host.now() - rows[-1]["t"]
    if age > max_age_s:
        return Check("meter csv", FAIL, f"latest row is {age:.0f} s old (> {max_age_s:g} s); is the meter logging?")
    if age < -max_age_s:
        return Check("meter csv", FAIL, f"latest row is {-age:.0f} s in the future; check meter clock / --tz")
    return Check("meter csv", PASS, f"latest row {age:.0f} s old, {len(rows)} rows")


def check_disk(db_path: str, models_dir: str, host: Host, warn_gb: float, fail_gb: float) -> list[Check]:
    out = []
    for label, p in (("disk[results]", Path(db_path).expanduser().resolve().parent),
                     ("disk[models]", Path(models_dir).expanduser())):
        probe = p
        while not host.exists(str(probe)) and probe != probe.parent:
            probe = probe.parent
        gb = host.disk_free(str(probe)) / 1e9
        status = FAIL if gb < fail_gb else WARN if gb < warn_gb else PASS
        out.append(Check(label, status, f"{gb:.1f} GB free at {probe}"))
    return out


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def check_db(db_path: str) -> tuple[list[Check], sqlite3.Connection | None]:
    p = Path(db_path).expanduser()
    if not p.exists():
        parent = p.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        ok = os.access(parent, os.W_OK)
        return [Check("results db", PASS if ok else FAIL, f"{p} absent; {parent} {'is' if ok else 'is NOT'} writable")], None
    # SQLite silently opens a read-only file read-only, so check write permission (file and WAL directory) first.
    if not (os.access(p, os.W_OK) and os.access(p.parent, os.W_OK)):
        return [Check("results db", FAIL, f"{p} or its directory is not writable")], None
    try:
        conn = sqlite3.connect(f"file:{p}?mode=rw", uri=True)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
    except sqlite3.Error as exc:
        return [Check("results db", FAIL, f"{p} not writable: {exc}")], None
    mock = 0
    for t in ("configs", "runs", "training_runs", "energy_blocks"):
        if _has_table(conn, t):
            mock += conn.execute(f"SELECT COUNT(*) FROM {t} WHERE is_mock=1").fetchone()[0]
    out = [Check("results db", PASS, f"{p} writable")]
    out.append(Check("mock rows", WARN if mock else PASS,
                     f"{mock} is_mock rows; report/analyze will refuse this DB" if mock else "none"))
    return out, conn


def run_preflight(dev: dict, configs: list[ExpConfig], db_path: str, models_dir: str, host: Host | None = None,
                  meter_csv: str | None = None, tz: str | None = None, verify_sha: bool = True,
                  max_temp_c: float = 50.0, meter_max_age_s: float = 60.0, warn_gb: float = 20.0,
                  fail_gb: float = 2.0) -> list[Check]:
    host = host or Host()
    checks = check_libraries(dev, host)
    checks += check_server(dev, host)
    db_checks, conn = check_db(db_path)
    checks += check_models(configs, conn, host, verify_sha)
    checks.append(check_governor(dev, host))
    checks += check_pi(dev, host, max_temp_c)
    checks.append(check_rapl(dev, host))
    checks.append(check_nvml(dev, host))
    checks.append(check_meter_csv(dev, host, meter_csv, tz, meter_max_age_s))
    checks += check_disk(db_path, models_dir, host, warn_gb, fail_gb)
    checks += db_checks
    if conn is not None:
        conn.close()
    return checks


def format_table(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks)
    lines = [f"{'CHECK'.ljust(w)}  STATUS  DETAIL", f"{'-' * w}  ------  ------"]
    lines += [f"{c.name.ljust(w)}  {c.status.ljust(6)}  {c.detail}" for c in checks]
    n = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL)}
    lines.append(f"\n{n[PASS]} PASS, {n[WARN]} WARN, {n[FAIL]} FAIL")
    return "\n".join(lines)
