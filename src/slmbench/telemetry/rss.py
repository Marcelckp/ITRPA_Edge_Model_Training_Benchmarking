"""Inference-server process memory: PSS when the kernel exposes it, else RSS, including child processes."""

from __future__ import annotations

import logging
import time
from typing import Callable

from .base import Reading, Source

log = logging.getLogger(__name__)


class ProcessMemorySource(Source):
    name = "psutil"
    provides = ("mem_mb",)

    def __init__(self, pid_getter: Callable[[], int | None]):
        self.pid_getter = pid_getter
        self.mem_kind = "PSS"
        try:
            import psutil  # lazy: [edge] extra

            self._ps = psutil
        except ImportError:
            self._ps = None
            log.info("psutil not installed; process memory unavailable")

    def available(self) -> bool:
        return self._ps is not None

    def _one(self, proc) -> float:
        if self.mem_kind == "PSS":
            try:
                return float(proc.memory_full_info().pss)
            except (AttributeError, self._ps.AccessDenied):
                self.mem_kind = "RSS"
        return float(proc.memory_info().rss)

    def read(self) -> Reading:
        pid = self.pid_getter()
        if pid is None:
            return Reading(time.time())
        proc = self._ps.Process(pid)
        total = self._one(proc)
        for ch in proc.children(recursive=True):
            try:
                total += self._one(ch)
            except self._ps.NoSuchProcess:
                pass
        return Reading(time.time(), mem_mb=total / (1024 * 1024))
