"""Ollama `/api/generate` streaming client. Convenience only; not used for measurement (DESIGN.md 3.0)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Iterable

from .base import Backend, BackendError, GenerationResult

log = logging.getLogger(__name__)


def parse_ollama_stream(lines: Iterable[bytes | str], t_start: float,
                        clock: Callable[[], float] = time.perf_counter) -> GenerationResult:
    parts: list[str] = []
    t_first: float | None = None
    final: dict[str, Any] = {}
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("error"):
            raise BackendError(str(obj["error"]))
        piece = obj.get("response", "")
        if piece:
            if t_first is None:
                t_first = clock()
            parts.append(piece)
        if obj.get("done"):
            final = obj
            break
    e2e = clock() - t_start
    eval_count = final.get("eval_count")
    eval_ns = final.get("eval_duration")
    prompt_ns = final.get("prompt_eval_duration")
    return GenerationResult(
        text="".join(parts), e2e_s=e2e, ttft_s=(t_first - t_start) if t_first is not None else None,
        tpot_s=(eval_ns / 1e9 / eval_count) if eval_ns and eval_count else None,
        prompt_tokens=final.get("prompt_eval_count"), output_tokens=eval_count,
        server_prompt_ms=prompt_ns / 1e6 if prompt_ns is not None else None,
        server_predicted_ms=eval_ns / 1e6 if eval_ns is not None else None,
    )


class OllamaBackend(Backend):
    kind = "ollama"

    def __init__(self, settings: dict[str, Any] | None = None, session: Any = None):
        super().__init__(settings)
        self.base_url = str(self.settings.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = float(self.settings.get("timeout_s", 600))
        self._session = session
        self.model_tag = ""

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def load(self, cfg: Any) -> None:
        tmpl = self.settings.get("model_template", "slmbench-{model}-{strategy}-{precision}")
        self.model_tag = tmpl.format(model=cfg.model, strategy=cfg.strategy, precision=cfg.precision,
                                     train_dataset=cfg.train_dataset or "none").lower()

    def generate(self, messages, max_new_tokens, temperature=0.0, seed=42) -> GenerationResult:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        body = {"model": self.model_tag, "system": system, "prompt": prompt, "stream": True,
                "options": {"temperature": temperature, "seed": seed, "num_predict": max_new_tokens,
                            "num_ctx": int(self.settings.get("num_ctx", 4096))}}
        t0 = time.perf_counter()
        resp = self.session.post(f"{self.base_url}/api/generate", json=body, stream=True, timeout=self.timeout)
        if resp.status_code != 200:
            raise BackendError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return parse_ollama_stream(resp.iter_lines(), t0)
        finally:
            resp.close()

    def health(self) -> bool:
        return self.session.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
