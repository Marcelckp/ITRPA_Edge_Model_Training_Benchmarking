"""NVIDIA Jetson board input power from a long-running `tegrastats` subprocess."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Sequence

from .base import Reading, Source

log = logging.getLogger(__name__)

_POWER_RE = re.compile(r"\b(VDD_IN|POM_5V_IN|VIN_SYS_5V0)\s+(\d+)mW(?:/\d+mW)?")
_TEMP_RE = re.compile(r"\b(?:CPU|cpu)@([\d.]+)C")


def parse_tegrastats(line: str) -> tuple[float | None, float | None]:
    """(input power W, CPU temperature C) from one tegrastats line."""
    p = _POWER_RE.search(line)
    t = _TEMP_RE.search(line)
    return (int(p.group(2)) / 1000.0 if p else None, float(t.group(1)) if t else None)


class TegrastatsSource(Source):
    name = "tegrastats"
    provides = ("power_w", "temp_c")

    def __init__(self, cmd: Sequence[str] = ("tegrastats", "--interval", "100")):
        self.cmd = list(cmd)
        self._last = Reading(time.time())
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def available(self) -> bool:
        return shutil.which(self.cmd[0]) is not None

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            w, c = parse_tegrastats(line)
            self._last = Reading(time.time(), power_w=w, temp_c=c)

    def read(self) -> Reading:
        if self._proc is None:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, text=True)
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self._last

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
