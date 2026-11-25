"""Chat-template parity: the training renderer must produce the same prompt llama-server builds."""

from __future__ import annotations

import difflib
from typing import Any


def _strip_bos(text: str, bos: str | None) -> str:
    return text[len(bos):] if bos and text.startswith(bos) else text


def compare(local: str, server: str, bos: str | None = None) -> dict[str, Any]:
    """BOS is stripped from both sides: HF templates write it as text, llama-server may add it as a token."""
    a, b = _strip_bos(local, bos), _strip_bos(server, bos)
    diff = "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "training-render", "llama-server", lineterm=""))
    return {"match": a == b, "local": a, "server": b, "diff": diff}
