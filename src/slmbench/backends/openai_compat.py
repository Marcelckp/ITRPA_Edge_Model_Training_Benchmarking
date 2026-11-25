"""OpenAI-compatible streaming chat client for llama.cpp `llama-server` (primary) and vLLM (reference)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Iterable

from .base import Backend, BackendError, GenerationResult, ServerProcess, format_cmd

log = logging.getLogger(__name__)


def parse_sse_stream(lines: Iterable[bytes | str], t_start: float,
                     clock: Callable[[], float] = time.perf_counter) -> GenerationResult:
    """Consume an SSE chat-completion stream; client TTFT is the arrival of the first non-empty delta."""
    text_parts: list[str] = []
    t_first: float | None = None
    n_chunks = 0
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            log.debug("skipping non-JSON SSE line: %s", payload[:80])
            continue
        if obj.get("error"):
            raise BackendError(str(obj["error"]))
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ch.get("text") or ""
            if piece:
                if t_first is None:
                    t_first = clock()
                n_chunks += 1
                text_parts.append(piece)
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
    t_end = clock()
    e2e = t_end - t_start
    out_tokens = timings.get("predicted_n") or usage.get("completion_tokens") or (n_chunks or None)
    prompt_tokens = timings.get("prompt_n") or usage.get("prompt_tokens")
    ttft = (t_first - t_start) if t_first is not None else None
    tpot = None
    if t_first is not None and out_tokens and out_tokens > 1:
        tpot = (t_end - t_first) / (out_tokens - 1)
    return GenerationResult(
        text="".join(text_parts), e2e_s=e2e, ttft_s=ttft, tpot_s=tpot,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        output_tokens=int(out_tokens) if out_tokens is not None else None,
        server_prompt_ms=float(timings["prompt_ms"]) if "prompt_ms" in timings else None,
        server_predicted_ms=float(timings["predicted_ms"]) if "predicted_ms" in timings else None,
    )


class OpenAICompatBackend(Backend):
    kind = "openai"

    def __init__(self, settings: dict[str, Any] | None = None, session: Any = None):
        super().__init__(settings)
        self.base_url = str(self.settings.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.timeout = float(self.settings.get("timeout_s", 600))
        self._session = session
        self.model_name = self.settings.get("served_model_name", "slmbench")
        self.template_kwargs = dict(self.settings.get("chat_template_kwargs") or {})

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests  # lazy: [edge] extra

            self._session = requests.Session()
        return self._session

    def request_body(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float,
                     seed: int) -> dict[str, Any]:
        return {"model": self.model_name, "messages": messages, "max_tokens": max_new_tokens,
                "temperature": temperature, "seed": seed, "stream": True,
                "stream_options": {"include_usage": True},
                **({"chat_template_kwargs": self.template_kwargs} if self.template_kwargs else {})}

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0,
                 seed: int = 42) -> GenerationResult:
        body = self.request_body(messages, max_new_tokens, temperature, seed)
        t0 = time.perf_counter()
        resp = self.session.post(f"{self.base_url}/v1/chat/completions", json=body, stream=True, timeout=self.timeout)
        if resp.status_code != 200:
            raise BackendError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return parse_sse_stream(resp.iter_lines(), t0)
        finally:
            resp.close()

    def health(self) -> bool:
        r = self.session.get(f"{self.base_url}/health", timeout=5)
        return r.status_code == 200

    def load(self, cfg: Any) -> None:
        if not self.settings.get("manage_server"):
            return
        if self.server and getattr(self, "_loaded_path", None) == cfg.model_path:
            return
        self.unload()
        paths = self.settings.get("_device_paths", {})
        cmd = format_cmd(self.settings["server_cmd"], model_path=cfg.model_path, **paths)
        self.server = ServerProcess(cmd, self.health, float(self.settings.get("startup_timeout_s", 300)),
                                    self.settings.get("server_log"))
        self.server.start()
        self._loaded_path = cfg.model_path


class LlamaCppBackend(OpenAICompatBackend):
    kind = "llamacpp"

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        """The prompt llama-server would build (needs --jinja to use the GGUF's own template)."""
        body: dict[str, Any] = {"messages": messages}
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        r = self.session.post(f"{self.base_url}/apply-template", json=body, timeout=30)
        if r.status_code != 200:
            raise BackendError(f"/apply-template HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["prompt"]

    def request_body(self, messages, max_new_tokens, temperature, seed):
        body = super().request_body(messages, max_new_tokens, temperature, seed)
        # Prompt-prefix reuse would shrink prefill time for repeated system prompts and bias TTFT.
        body["cache_prompt"] = False
        body["timings_per_token"] = False
        return body

    def count_tokens(self, text: str) -> int | None:
        try:
            r = self.session.post(f"{self.base_url}/tokenize", json={"content": text}, timeout=30)
            if r.status_code == 200:
                return len(r.json().get("tokens", []))
        except Exception as exc:
            log.debug("tokenize failed: %s", exc)
        return None


class VLLMBackend(OpenAICompatBackend):
    kind = "vllm"

    def count_tokens(self, text: str) -> int | None:
        try:
            r = self.session.post(f"{self.base_url}/tokenize", json={"model": self.model_name, "prompt": text},
                                  timeout=30)
            if r.status_code == 200:
                return int(r.json().get("count"))
        except Exception as exc:
            log.debug("tokenize failed: %s", exc)
        return None
