"""NVIDIA GPU power, VRAM, temperature and total-energy counter via NVML (board power only)."""

from __future__ import annotations

import logging
import time

from .base import Reading, Source

log = logging.getLogger(__name__)


class NvmlSource(Source):
    name = "nvml"
    provides = ("power_w", "vram_mb", "temp_c")
    mem_kind = "VRAM"

    def __init__(self, index: int = 0):
        self.index = index
        self._nv = None
        self._h = None
        self._counter_ok = False
        try:
            import pynvml  # lazy: [gpu] extra

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
                self._counter_ok = True
            except Exception:
                log.info("NVML total-energy counter not supported; using 10 ms power sampling")
        except Exception as exc:
            log.info("NVML unavailable: %s", exc)

    def available(self) -> bool:
        return self._h is not None

    @property
    def has_counter(self) -> bool:
        return self._counter_ok

    def read(self) -> Reading:
        nv, h = self._nv, self._h
        power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        mem = nv.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
        temp = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        return Reading(time.time(), power_w=power, vram_mb=mem, temp_c=temp)

    def energy_j(self) -> float | None:
        if not self._counter_ok:
            return None
        return self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h) / 1000.0

    def close(self) -> None:
        if self._nv is not None:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass
