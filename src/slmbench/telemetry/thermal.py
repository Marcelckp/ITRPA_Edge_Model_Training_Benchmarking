"""CPU/SoC temperature from /sys/class/thermal."""

from __future__ import annotations

import time
from pathlib import Path

from .base import Reading, Source


class ThermalZoneSource(Source):
    name = "thermal-zone"
    provides = ("temp_c",)

    def __init__(self, zone: str = "thermal_zone0", root: Path = Path("/sys/class/thermal")):
        self.path = root / zone / "temp"

    def available(self) -> bool:
        return self.path.exists()

    def read(self) -> Reading:
        return Reading(time.time(), temp_c=int(self.path.read_text().strip()) / 1000.0)
