"""Record the software and hardware environment of a measurement host."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .prompts import PROMPT_VERSION

log = logging.getLogger(__name__)

LIBRARIES = ("slmbench", "torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "pynvml",
             "psutil", "requests", "rank_bm25", "numpy", "scipy", "pandas", "matplotlib", "vllm", "pyyaml")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd: list[str]) -> str | None:
    if shutil.which(os.path.expanduser(cmd[0])) is None and not Path(os.path.expanduser(cmd[0])).exists():
        return None
    try:
        r = subprocess.run([os.path.expanduser(c) for c in cmd], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip()[:500] or None
    except Exception as exc:
        log.debug("%s failed: %s", cmd, exc)
        return None


def cpu_model() -> str | None:
    """x86 lists a numeric "model" before "model name", so the named keys win."""
    fields: dict[str, str] = {}
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        key, _, val = line.partition(":")
        fields.setdefault(key.strip().lower(), val.strip())
    for key in ("model name", "hardware", "model"):
        if fields.get(key):
            return fields[key]
    return platform.processor() or None


def mem_total_mb() -> float | None:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024
    return None


def collect(device: str, llama_cpp_dir: str | None = None, note: str | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "hostname": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(),
        "machine": platform.machine(), "cpu": cpu_model(), "cpu_count": os.cpu_count(), "ram_mb": mem_total_mb(),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_freq_khz": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "slmbench_version": __version__, "prompt_version": PROMPT_VERSION,
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,persistence_mode,clocks.max.sm",
                     "--format=csv,noheader"]),
        "pi_model": _read("/proc/device-tree/model"),
        "pi_throttled": _run(["vcgencmd", "get_throttled"]),
    }
    for lib in LIBRARIES:
        try:
            env[f"lib/{lib}"] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env[f"lib/{lib}"] = None
    if llama_cpp_dir:
        env["llama_cpp_version"] = _run([str(Path(llama_cpp_dir) / "build" / "bin" / "llama-server"), "--version"])
        # Container images ship llama.cpp without .git and record the pinned commit in the environment instead.
        commit = _run(["git", "-C", os.path.expanduser(llama_cpp_dir), "rev-parse", "HEAD"]) or ""
        is_sha = len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit)
        env["llama_cpp_commit"] = commit if is_sha else os.environ.get("LLAMA_CPP_COMMIT")
        env["llama_cpp_tag"] = os.environ.get("LLAMA_CPP_TAG")
    if note:
        env["note"] = note
    return {f"{device}/{k}": v for k, v in env.items()}
