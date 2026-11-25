"""Intel/AMD RAPL CPU-package energy via /sys/class/powercap (package energy, not wall power)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .base import Reading, Source

log = logging.getLogger(__name__)

POWERCAP = Path("/sys/class/powercap")


class RaplSource(Source):
    name = "rapl-package"
    provides = ("power_w",)

    def __init__(self, zone: str = "intel-rapl:0", root: Path = POWERCAP):
        self.dir = root / zone
        self._max = None
        self._last_raw: int | None = None
        self._acc_uj = 0
        self._lock = threading.Lock()
        self._prev: tuple[float, float] | None = None
        try:
            self._max = int((self.dir / "max_energy_range_uj").read_text())
            self._read_raw()
        except (OSError, ValueError) as exc:
            log.info("RAPL unavailable at %s: %s (reading usually needs root)", self.dir, exc)
            self._max = None

    def available(self) -> bool:
        return self._max is not None

    @property
    def has_counter(self) -> bool:
        return self.available()

    def _read_raw(self) -> int:
        return int((self.dir / "energy_uj").read_text())

    def energy_j(self) -> float | None:
        """Monotonic cumulative energy since construction; handles counter wrap-around."""
        if self._max is None:
            return None
        with self._lock:
            raw = self._read_raw()
            if self._last_raw is not None:
                delta = raw - self._last_raw
                if delta < 0:
                    delta += self._max + 1
                self._acc_uj += delta
            self._last_raw = raw
            return self._acc_uj / 1e6

    def read(self) -> Reading:
        now = time.time()
        e = self.energy_j()
        power = None
        if self._prev is not None and e is not None and now > self._prev[0]:
            power = (e - self._prev[1]) / (now - self._prev[0])
        self._prev = (now, e) if e is not None else self._prev
        return Reading(now, power_w=power)
