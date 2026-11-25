"""Telemetry sources, a background sampler thread, and per-query/per-block measurement windows."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


log = logging.getLogger(__name__)


@dataclass
class Reading:
    t: float
    power_w: float | None = None
    vram_mb: float | None = None
    mem_mb: float | None = None
    temp_c: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"t": self.t, "power_w": self.power_w, "vram_mb": self.vram_mb, "mem_mb": self.mem_mb,
                "temp_c": self.temp_c}


class Source:
    """A telemetry source. ``read`` returns a partial Reading; ``energy_j`` a cumulative counter or None."""

    name = "source"
    provides: tuple[str, ...] = ()
    mem_kind: str | None = None
    min_interval_s = 0.0

    def available(self) -> bool:
        return False

    def read(self) -> Reading:
        return Reading(time.time())

    def energy_j(self) -> float | None:
        return None

    @property
    def has_counter(self) -> bool:
        return False

    def close(self) -> None:
        pass


class EnergyAccumulator:
    """Running trapezoid integral of power, so a window of any length needs no stored samples."""

    def __init__(self, t0: float, last: Reading | None):
        self.t0 = t0
        self.energy_j = 0.0
        self.n = 0
        self._last = (t0, last.power_w) if last is not None and last.power_w is not None else None

    def add(self, t: float, p: float | None) -> None:
        if p is None or t < self.t0:
            return
        if self._last is None:
            self._last = (self.t0, p)
        lt, lp = self._last
        self.energy_j += (t - lt) * (p + lp) / 2
        self._last = (t, p)
        self.n += 1

    def close(self, t1: float) -> float | None:
        if self._last is None:
            return None
        lt, lp = self._last
        return self.energy_j + max(0.0, t1 - lt) * lp


class Sampler:
    """Polls one source on a daemon thread. Errors are logged once and the sample skipped."""

    def __init__(self, source: Source, interval_s: float, keep_s: float = 3600.0):
        self.source = source
        self.interval_s = max(interval_s, source.min_interval_s)
        self.keep_s = keep_s
        self.samples: list[Reading] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._accs: dict[int, EnergyAccumulator] = {}
        self._next_acc = 0

    def open_accumulator(self, t0: float) -> int:
        with self._lock:
            last = self.samples[-1] if self.samples else None
            key = self._next_acc
            self._next_acc += 1
            self._accs[key] = EnergyAccumulator(t0, last)
        return key

    def close_accumulator(self, key: int, t1: float) -> float | None:
        with self._lock:
            acc = self._accs.pop(key, None)
        return acc.close(t1) if acc else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                r = self.source.read()
                with self._lock:
                    self.samples.append(r)
                    for acc in self._accs.values():
                        acc.add(r.t, r.power_w)
                    cutoff = r.t - self.keep_s
                    if self.samples and self.samples[0].t < cutoff:
                        self.samples = [s for s in self.samples if s.t >= cutoff]
            except Exception as exc:
                if not self._warned:
                    log.warning("%s read failed: %s", self.source.name, exc)
                    self._warned = True
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=f"sampler-{self.source.name}", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def window(self, t0: float, t1: float, pad_s: float = 0.0) -> list[Reading]:
        with self._lock:
            inside = [s for s in self.samples if t0 - pad_s <= s.t <= t1 + pad_s]
            if not inside:
                before = [s for s in self.samples if s.t <= t1]
                inside = before[-1:]
        return inside


@dataclass
class WindowStats:
    energy_j: float | None = None
    avg_power_w: float | None = None
    peak_mem_mb: float | None = None
    mem_kind: str | None = None
    peak_vram_mb: float | None = None
    temp_c: float | None = None
    energy_source: str | None = None
    samples: list[Reading] | None = None
    cpu_s: float | None = None
    wall_s: float | None = None


@dataclass
class Mark:
    t: float
    counter_j: float | None
    acc_key: int | None = None
    cpu_s: float | None = None


class TelemetrySession:
    """Runs samplers for power, memory and temperature; measures windows between ``begin`` and ``end``."""

    def __init__(self, power: Source | None = None, memory: Iterable[Source] = (), temperature: Source | None = None,
                 interval_s: float = 0.1, keep_raw: bool = False, cpu: Any = None):
        self.cpu = cpu
        self.power = power
        self.memory = [m for m in memory if m is not None]
        self.temperature = temperature
        self.keep_raw = keep_raw
        self._samplers: dict[str, Sampler] = {}
        for role, src in [("power", power), ("temperature", temperature)] + [(f"mem{i}", m) for i, m in enumerate(self.memory)]:
            if src is not None:
                self._samplers[role] = Sampler(src, interval_s)

    @property
    def energy_source(self) -> str | None:
        if self.power is None:
            return None
        return f"{self.power.name}-counter" if self.power.has_counter else f"{self.power.name}-trapezoid"

    def start(self) -> None:
        for s in self._samplers.values():
            s.start()

    def stop(self) -> None:
        for s in self._samplers.values():
            s.stop()
        for src in [self.power, self.temperature, *self.memory]:
            if src is not None:
                src.close()

    def begin(self) -> Mark:
        c = self.power.energy_j() if self.power is not None and self.power.has_counter else None
        t = time.time()
        key = self._samplers["power"].open_accumulator(t) if c is None and "power" in self._samplers else None
        return Mark(t, c, key, self._cpu_now())

    def _cpu_now(self) -> float | None:
        if self.cpu is None:
            return None
        try:
            return self.cpu.cpu_s()
        except (OSError, ValueError, KeyError):
            return None

    def end(self, mark: Mark) -> WindowStats:
        t1 = time.time()
        c1 = self.power.energy_j() if self.power is not None and self.power.has_counter else None
        st = WindowStats(energy_source=self.energy_source, wall_s=t1 - mark.t)
        c_now = self._cpu_now() if mark.cpu_s is not None else None
        if c_now is not None:
            st.cpu_s = c_now - mark.cpu_s
        dur = max(t1 - mark.t, 1e-9)
        if mark.counter_j is not None and c1 is not None:
            st.energy_j = c1 - mark.counter_j
            st.avg_power_w = st.energy_j / dur
        elif mark.acc_key is not None:
            st.energy_j = self._samplers["power"].close_accumulator(mark.acc_key, t1)
            st.avg_power_w = st.energy_j / dur if st.energy_j is not None else None
        raw: list[Reading] = []
        for i, m in enumerate(self.memory):
            win = self._samplers[f"mem{i}"].window(mark.t, t1)
            raw.extend(win)
            vals = [r.mem_mb for r in win if r.mem_mb is not None]
            if vals and st.peak_mem_mb is None:
                st.peak_mem_mb, st.mem_kind = max(vals), m.mem_kind
            v = [r.vram_mb for r in win if r.vram_mb is not None]
            if v:
                st.peak_vram_mb = max(v) if st.peak_vram_mb is None else max(st.peak_vram_mb, max(v))
        if "temperature" in self._samplers:
            temps = [r.temp_c for r in self._samplers["temperature"].window(mark.t, t1) if r.temp_c is not None]
            st.temp_c = max(temps) if temps else None
        if self.keep_raw:
            raw.extend(self._samplers["power"].window(mark.t, t1) if "power" in self._samplers else [])
            st.samples = raw
        return st

    def measure_idle(self, seconds: float) -> tuple[float | None, str | None]:
        """Mean power over ``seconds`` while nothing is generating."""
        if self.power is None:
            return None, None
        mark = self.begin()
        time.sleep(seconds)
        st = self.end(mark)
        return st.avg_power_w, self.energy_source

    def current_temp(self) -> float | None:
        if self.temperature is None:
            return None
        try:
            return self.temperature.read().temp_c
        except Exception:
            return None
