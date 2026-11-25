"""Container CPU time and memory from the cgroup filesystem (v1 and v2), plus joins of 1 Hz host samples.

CPU time feeds the Green Algorithms estimate in ``slmbench.energy_estimate``; it is not an energy measurement.
Inside a Docker container the container's own cgroup is mounted at the root of ``/sys/fs/cgroup``.
"""

from __future__ import annotations

import bisect
import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .base import Reading, Source

log = logging.getLogger(__name__)


@dataclass
class CgroupPaths:
    version: int
    cpu: Path
    memory: Path
    cpu_stat: Path | None = None


@dataclass
class CgroupReading:
    t: float
    cpu_ns: int
    mem_bytes: int | None
    mem_peak_bytes: int | None


def parse_kv(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def parse_v2_cpu_ns(cpu_stat: str) -> int:
    """cgroup v2 ``cpu.stat`` reports ``usage_usec``."""
    return parse_kv(cpu_stat)["usage_usec"] * 1000


def parse_int(text: str) -> int | None:
    text = text.strip()
    return None if not text or text == "max" else int(text)


def _proc_paths(proc_cgroup: str) -> dict[str, str]:
    """Controller -> cgroup path from /proc/self/cgroup ('' key for the v2 unified hierarchy)."""
    out: dict[str, str] = {}
    for line in proc_cgroup.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrls, path = parts
        for c in (ctrls.split(",") if ctrls else [""]):
            out[c] = path
    return out


def find_cgroup(root: str | os.PathLike = "/sys/fs/cgroup", proc_cgroup: str | None = None) -> CgroupPaths | None:
    """Locate this process's cgroup counters; tries the /proc/self/cgroup path, then the mount root."""
    root = Path(root)
    if proc_cgroup is None:
        try:
            proc_cgroup = Path("/proc/self/cgroup").read_text()
        except OSError:
            proc_cgroup = ""
    paths = _proc_paths(proc_cgroup)
    if (root / "cgroup.controllers").exists():
        rel = paths.get("", "/").lstrip("/")
        for d in ([root / rel] if rel else []) + [root]:
            if (d / "cpu.stat").exists():
                return CgroupPaths(2, d / "cpu.stat", d / "memory.current", d / "cpu.stat")
        return None
    cpu_rel = paths.get("cpuacct", paths.get("cpu", "/")).lstrip("/")
    mem_rel = paths.get("memory", "/").lstrip("/")
    for cpu_base in ("cpuacct", "cpu,cpuacct", "cpu"):
        for rel_c, rel_m in ((cpu_rel, mem_rel), ("", "")):
            c = root / cpu_base / rel_c / "cpuacct.usage"
            if c.exists():
                m = root / "memory" / rel_m
                stat = root / "cpu" / rel_c / "cpu.stat"
                return CgroupPaths(1, c, m, stat if stat.exists() else None)
    return None


def read_cgroup(p: CgroupPaths) -> CgroupReading:
    t = time.time()
    if p.version == 1:
        cpu = int(p.cpu.read_text())
        mem = _maybe_int(p.memory / "memory.usage_in_bytes")
        peak = _maybe_int(p.memory / "memory.max_usage_in_bytes")
    else:
        cpu = parse_v2_cpu_ns(p.cpu.read_text())
        mem = _maybe_int(p.memory)
        peak = _maybe_int(p.memory.parent / "memory.peak")
    return CgroupReading(t, cpu, mem, peak)


def _maybe_int(path: Path) -> int | None:
    try:
        return parse_int(path.read_text())
    except (OSError, ValueError):
        return None


class CgroupSource(Source):
    """Telemetry source: container memory (mem_kind 'cgroup') and cumulative CPU seconds via ``cpu_s``."""

    name = "cgroup"
    provides = ("mem_mb",)
    mem_kind = "cgroup"

    def __init__(self, root: str | os.PathLike = "/sys/fs/cgroup", proc_cgroup: str | None = None):
        self.paths = find_cgroup(root, proc_cgroup)

    def available(self) -> bool:
        return self.paths is not None

    def read(self) -> Reading:
        r = read_cgroup(self.paths)
        return Reading(r.t, mem_mb=None if r.mem_bytes is None else r.mem_bytes / (1024 * 1024))

    def cpu_s(self) -> float:
        return read_cgroup(self.paths).cpu_ns / 1e9


@dataclass
class HostSample:
    t: float
    cpu_ns: int
    mem_bytes: int | None
    container_id: str


def read_host_csv(path: str | os.PathLike) -> list[HostSample]:
    """Rows written by scripts/pilot/cgroup_logger.py, sorted by time."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("utc") == "utc" or not r.get("cpu_usage_ns"):
                continue
            mem = r.get("mem_usage_bytes")
            out.append(HostSample(float(r["unix_s"]), int(r["cpu_usage_ns"]), int(mem) if mem else None,
                                  r.get("container_id", "")))
    out.sort(key=lambda s: s.t)
    return out


def iso_to_unix(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


@dataclass
class WindowUsage:
    wall_s: float
    cpu_s: float | None
    peak_mem_mb: float | None
    n_samples: int
    coverage: str


def window_usage(samples: Sequence[HostSample], t0: float, t1: float) -> WindowUsage:
    """CPU seconds in [t0, t1] by linear interpolation of the cumulative counter between 1 Hz samples.

    Only samples of one container instance are interpolated across; a window that is not inside the sampled
    span returns cpu_s None. Peak memory is the largest sample in the window (or the nearest one before it).
    """
    wall = max(0.0, t1 - t0)
    times = [s.t for s in samples]
    i0 = bisect.bisect_right(times, t0) - 1
    i1 = bisect.bisect_left(times, t1)
    if t1 < t0 or i0 < 0 or i1 >= len(samples) or samples[i0].container_id != samples[i1].container_id:
        return WindowUsage(wall, None, None, 0, "outside sampled span")
    cid = samples[i0].container_id
    if any(s.container_id != cid for s in samples[i0:i1 + 1]):
        return WindowUsage(wall, None, None, 0, "container restarted inside window")
    c0, c1 = _counter_at(samples, times, t0), _counter_at(samples, times, t1)
    inside = [s.mem_bytes for s in samples[i0:i1 + 1] if s.mem_bytes is not None]
    peak = max(inside) / (1024 * 1024) if inside else None
    return WindowUsage(wall, max(0.0, (c1 - c0) / 1e9), peak, i1 - i0 + 1, "interpolated")


def _counter_at(samples: Sequence[HostSample], times: list[float], t: float) -> float:
    j = bisect.bisect_right(times, t)
    a = samples[j - 1]
    if j >= len(samples) or a.t == t:
        return float(a.cpu_ns)
    b = samples[j]
    return a.cpu_ns + (b.cpu_ns - a.cpu_ns) * (t - a.t) / (b.t - a.t)


def join_windows(samples: Sequence[HostSample],
                 windows: Iterable[tuple[str, float, float]]) -> list[tuple[str, WindowUsage]]:
    return [(ref, window_usage(samples, t0, t1)) for ref, t0, t1 in windows]
