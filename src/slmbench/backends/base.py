"""Backend interface and a loopback-only inference-server process manager."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class BackendError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    e2e_s: float
    ttft_s: float | None = None
    tpot_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    server_prompt_ms: float | None = None
    server_predicted_ms: float | None = None
    # Only the mock backend fills these: simulated (seconds since request start, watts) samples and memory.
    power_trace: list[tuple[float, float]] = field(default_factory=list)
    simulated_mem_mb: float | None = None

    @property
    def tokens_per_s(self) -> float | None:
        tpot = self.primary_tpot_s
        return 1.0 / tpot if tpot and tpot > 0 else None

    @property
    def primary_ttft_s(self) -> float | None:
        return self.server_prompt_ms / 1000.0 if self.server_prompt_ms is not None else self.ttft_s

    @property
    def primary_tpot_s(self) -> float | None:
        if self.server_predicted_ms is not None and self.output_tokens:
            return self.server_predicted_ms / 1000.0 / self.output_tokens
        return self.tpot_s


class Backend:
    kind = "base"
    is_mock = False

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})
        self.server: ServerProcess | None = None

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0,
                 seed: int = 42) -> GenerationResult:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int | None:
        return None

    def health(self) -> bool:
        return True

    def load(self, cfg: Any) -> None:
        """Make ``cfg.model_path`` the served model, starting a managed server if configured."""

    def unload(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None

    @property
    def server_pid(self) -> int | None:
        if self.server and self.server.proc:
            return self.server.proc.pid
        pidfile = self.settings.get("server_pidfile")
        if pidfile and Path(os.path.expanduser(pidfile)).exists():
            return int(Path(os.path.expanduser(pidfile)).read_text().strip())
        pid = self.settings.get("server_pid")
        return int(pid) if pid else None


def check_loopback_cmd(cmd: Sequence[str]) -> None:
    """Refuse to launch a server unless it is explicitly bound to a loopback address."""
    args = list(cmd)
    host = None
    for i, a in enumerate(args):
        if a == "--host" and i + 1 < len(args):
            host = args[i + 1]
        elif a.startswith("--host="):
            host = a.split("=", 1)[1]
    if host is None:
        raise BackendError("server_cmd must pass --host 127.0.0.1 explicitly")
    if host not in LOOPBACK:
        raise BackendError(f"server_cmd binds {host}; only loopback is allowed")


class ServerProcess:
    def __init__(self, cmd: Sequence[str], health: Callable[[], bool], startup_timeout_s: float = 300,
                 log_path: str | Path | None = None):
        check_loopback_cmd(cmd)
        self.cmd = list(cmd)
        self.health = health
        self.startup_timeout_s = startup_timeout_s
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self._log_fh: Any = None

    def start(self) -> None:
        log.info("starting server: %s", " ".join(self.cmd))
        out: Any = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "ab")
            out = self._log_fh
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BackendError(f"server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if self.health():
                    log.info("server ready (pid %d)", self.proc.pid)
                    return
            except Exception:
                pass
            time.sleep(1.0)
        self.stop()
        raise BackendError(f"server not healthy after {self.startup_timeout_s}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None


def format_cmd(cmd: Sequence[str], **fields: Any) -> list[str]:
    return [os.path.expanduser(str(a).format(**fields)) for a in cmd]
