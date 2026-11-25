"""Raspberry Pi 5 PMIC rail power (`vcgencmd pmic_read_adc`), throttle state and SoC temperature.

pmic_read_adc reports voltages/currents for some rails only, not board input power, so the sum is a
partial lower bound; the inline USB-C meter is primary (DESIGN.md 3.5).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time

from .base import Reading, Source

log = logging.getLogger(__name__)

_RAIL_RE = re.compile(r"^\s*(\S+?)_([AV])\s+(?:current|volt)\(\d+\)=([\d.]+)[AV]\s*$")
THROTTLE_CURRENT_MASK = 0xF


def parse_pmic(output: str) -> float | None:
    """Sum of V x A over rails that report both a current and a voltage."""
    amps: dict[str, float] = {}
    volts: dict[str, float] = {}
    for line in output.splitlines():
        m = _RAIL_RE.match(line)
        if not m:
            continue
        rail, kind, val = m.group(1), m.group(2), float(m.group(3))
        (amps if kind == "A" else volts)[rail] = val
    rails = set(amps) & set(volts)
    return sum(amps[r] * volts[r] for r in rails) if rails else None


def parse_throttled(output: str) -> int | None:
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", output)
    return int(m.group(1), 16) if m else None


def is_throttled_now(value: int | None) -> bool | None:
    return None if value is None else bool(value & THROTTLE_CURRENT_MASK)


def _vcgencmd(*args: str) -> str:
    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5, check=True).stdout


class PiPmicSource(Source):
    name = "pmic_partial_lower_bound"
    provides = ("power_w",)
    min_interval_s = 0.5  # vcgencmd is slow and perturbs the CPU; DESIGN.md caps it at 2 Hz

    def available(self) -> bool:
        return shutil.which("vcgencmd") is not None

    def read(self) -> Reading:
        return Reading(time.time(), power_w=parse_pmic(_vcgencmd("pmic_read_adc")))


def read_throttled() -> int | None:
    if shutil.which("vcgencmd") is None:
        return None
    try:
        return parse_throttled(_vcgencmd("get_throttled"))
    except Exception as exc:
        log.debug("get_throttled failed: %s", exc)
        return None
