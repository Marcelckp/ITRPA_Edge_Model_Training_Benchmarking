"""Inference backends. All default to 127.0.0.1."""

from __future__ import annotations

from typing import Any

from .base import Backend, BackendError, GenerationResult


def make_backend(kind: str, settings: dict[str, Any] | None = None) -> Backend:
    if kind == "mock":
        from .mock import MockBackend

        return MockBackend(settings)
    if kind == "llamacpp":
        from .openai_compat import LlamaCppBackend

        return LlamaCppBackend(settings)
    if kind == "vllm":
        from .openai_compat import VLLMBackend

        return VLLMBackend(settings)
    if kind == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend(settings)
    raise ValueError(f"unknown backend {kind}")


__all__ = ["Backend", "BackendError", "GenerationResult", "make_backend"]
