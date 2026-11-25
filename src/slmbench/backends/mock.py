"""Deterministic offline backend for tests. Everything it produces is simulated and stored with is_mock=1."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..prompts import heuristic_token_count
from .base import Backend, GenerationResult

_PRECISION_SPEED = {"F16": 1.0, "Q8_0": 0.7, "Q4_K_M": 0.5, "bf16": 0.3}
_PRECISION_MEM_MB = {"F16": 2400.0, "Q8_0": 1300.0, "Q4_K_M": 800.0, "bf16": 2400.0}


def _h(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class MockBackend(Backend):
    kind = "mock"
    is_mock = True

    def __init__(self, settings: dict[str, Any] | None = None):
        super().__init__(settings)
        self.cfg: Any = None

    def load(self, cfg: Any) -> None:
        self.cfg = cfg

    def count_tokens(self, text: str) -> int | None:
        return heuristic_token_count(text)

    def _answer(self, user: str, h: int) -> str:
        if "yes, no, or maybe" in user:
            return ["yes", "no", "maybe", "Yes.", "I think no"][h % 5]
        if "single number" in user:
            nums = re.findall(r"\d+(?:\.\d+)?", user)
            return nums[h % len(nums)] if nums and h % 3 else f"{(h % 1000) / 10:g}"
        if "letter of the correct option" in user:
            return "ABCD"[h % 4]
        return "ok"

    def generate(self, messages, max_new_tokens, temperature=0.0, seed=42) -> GenerationResult:
        user = messages[-1]["content"]
        c = self.cfg
        # Device is excluded so the same GGUF answers identically everywhere, as greedy decoding should.
        ident = (c.model, c.strategy, c.precision, c.train_dataset) if c else ("none",)
        h = _h(*ident, seed, user)
        text = self._answer(user, h)
        prompt_tokens = sum(heuristic_token_count(m["content"]) for m in messages)
        out_tokens = max(1, min(max_new_tokens, len(text.split()) + 1))
        speed = _PRECISION_SPEED.get(getattr(c, "precision", "F16"), 1.0)
        prefill_ms = prompt_tokens * 0.8 * speed + (h % 7)
        decode_ms = out_tokens * 40.0 * speed + (h % 5)
        e2e = (prefill_ms + decode_ms) / 1000.0 + 0.002
        base_w = 4.0 + 2.0 * speed
        trace = [(t * e2e / 9, base_w + ((h >> t) % 3) * 0.1) for t in range(10)]
        return GenerationResult(text=text, e2e_s=e2e, ttft_s=prefill_ms / 1000.0 + 0.001,
                                tpot_s=decode_ms / 1000.0 / out_tokens, prompt_tokens=prompt_tokens,
                                output_tokens=out_tokens, server_prompt_ms=prefill_ms,
                                server_predicted_ms=decode_ms, power_trace=trace,
                                simulated_mem_mb=_PRECISION_MEM_MB.get(getattr(c, "precision", "F16"), 1000.0)
                                + prompt_tokens * 0.01 + (h % 11))
