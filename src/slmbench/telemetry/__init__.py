"""Telemetry: build the sources named in a device profile, falling back gracefully when one is missing."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .base import Reading, Sampler, Source, TelemetrySession, WindowStats

log = logging.getLogger(__name__)


class MockPowerSource(Source):
    """Constant simulated power for idle-baseline tests with the mock backend."""

    name = "mock"
    provides = ("power_w", "temp_c")

    def __init__(self, watts: float = 3.0, temp_c: float = 40.0):
        self.watts, self.temp_c = watts, temp_c

    def available(self) -> bool:
        return True

    def read(self) -> Reading:
        return Reading(time.time(), power_w=self.watts, temp_c=self.temp_c)


def make_source(kind: str, settings: dict[str, Any], pid_getter: Callable[[], int | None]) -> Source | None:
    if kind == "nvml":
        from .nvml import NvmlSource

        src: Source = NvmlSource(int(settings.get("gpu_index", 0)))
    elif kind == "rss":
        from .rss import ProcessMemorySource

        src = ProcessMemorySource(pid_getter)
    elif kind == "rapl":
        from .rapl import RaplSource

        src = RaplSource(settings.get("rapl_zone", "intel-rapl:0"))
    elif kind == "pi_pmic":
        from .pi_pmic import PiPmicSource

        src = PiPmicSource()
    elif kind == "tegrastats":
        from .tegrastats import TegrastatsSource

        src = TegrastatsSource(settings.get("tegrastats_cmd", ["tegrastats", "--interval", "100"]))
    elif kind == "thermal_zone":
        from .thermal import ThermalZoneSource

        src = ThermalZoneSource(settings.get("thermal_zone", "thermal_zone0"))
    elif kind == "cgroup":
        from .cgroup import CgroupSource

        src = CgroupSource(settings.get("cgroup_root", "/sys/fs/cgroup"))
    elif kind == "mock":
        src = MockPowerSource()
    else:
        log.warning("unknown telemetry source %s", kind)
        return None
    if not src.available():
        log.warning("telemetry source %s unavailable on this host; continuing without it", kind)
        return None
    return src


def build_session(telemetry: dict[str, Any], pid_getter: Callable[[], int | None], mock: bool = False) -> TelemetrySession:
    if mock:
        return TelemetrySession(power=MockPowerSource(), temperature=MockPowerSource(),
                                interval_s=float(telemetry.get("interval_s", 0.05)))
    power = None
    for kind in telemetry.get("power", []):
        power = make_source(kind, telemetry, pid_getter)
        if power:
            break
    memory = [m for m in (make_source(k, telemetry, pid_getter) for k in telemetry.get("memory", [])) if m]
    temp_kind = telemetry.get("temperature")
    temperature = make_source(temp_kind, telemetry, pid_getter) if temp_kind else None
    cpu = None
    if telemetry.get("cpu") == "cgroup":
        from .cgroup import CgroupSource

        cpu = CgroupSource(telemetry.get("cgroup_root", "/sys/fs/cgroup"))
        if not cpu.available():
            log.warning("telemetry source cgroup unavailable on this host; no CPU time recorded")
            cpu = None
    return TelemetrySession(power, memory, temperature, float(telemetry.get("interval_s", 0.1)), cpu=cpu)


__all__ = ["Reading", "Sampler", "Source", "TelemetrySession", "WindowStats", "MockPowerSource",
           "make_source", "build_session"]
