"""Fixed prompt templates shared by training and evaluation."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)

SYSTEM = {
    "pubmedqa": "You are a biomedical question answering assistant.",
    "finqa": "You are a financial question answering assistant.",
    "arc_easy": "You are a science question answering assistant.",
}
INSTRUCTION = {
    "pubmedqa": "Answer with exactly one word: yes, no, or maybe.",
    "finqa": "Answer with a single number only.",
    "arc_easy": "Answer with the letter of the correct option only.",
}
PROMPT_VERSION = hashlib.sha256(repr((SYSTEM, INSTRUCTION)).encode()).hexdigest()[:12]

# Rough upper bound used only when no tokenizer is available (mock backend, tests).
CHARS_PER_TOKEN = 3.0
TEMPLATE_OVERHEAD_TOKENS = 64


@dataclass
class Prompt:
    messages: list[dict[str, str]]
    truncated: bool = False
    context_units: list[str] = field(default_factory=list)


def heuristic_token_count(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def serialise_table_row(header: Sequence[str], row: Sequence[str]) -> str:
    if not header or row is header:
        return " | ".join(row)
    label = row[0] if row else ""
    cells = "; ".join(f"{h} = {c}" for h, c in zip(header[1:], row[1:]))
    return f"{label}: {cells}" if cells else label


def finqa_units(item: dict[str, Any]) -> list[tuple[str, str]]:
    """Document units with FinQA gold_inds ids: text_i over pre_text + post_text, table_i over table rows."""
    units: list[tuple[str, str]] = []
    texts = list(item.get("pre_text", [])) + list(item.get("post_text", []))
    n_pre = len(item.get("pre_text", []))
    table = item.get("table", [])
    header = table[0] if table else []
    pre = [(f"text_{i}", t) for i, t in enumerate(texts[:n_pre])]
    post = [(f"text_{n_pre + i}", t) for i, t in enumerate(texts[n_pre:])]
    rows = [(f"table_{i}", " | ".join(r) if i == 0 else serialise_table_row(header, r)) for i, r in enumerate(table)]
    units.extend(pre)
    units.extend(rows)
    units.extend(post)
    return units


def fit_units(units: Sequence[str], budget_tokens: int, count: Callable[[str], int]) -> tuple[list[str], bool]:
    """Keep units in document order until the token budget is spent; report whether any were dropped."""
    kept, used = [], 0
    for u in units:
        n = count(u) + 1
        if used + n > budget_tokens:
            return kept, True
        kept.append(u)
        used += n
    return kept, False


def _user_text(dataset: str, item: dict[str, Any], context: str | None) -> str:
    parts = []
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"Question: {item['question']}")
    if dataset == "arc_easy":
        opts = "\n".join(f"{c['label']}. {c['text']}" for c in item["choices"])
        parts.append(f"Options:\n{opts}")
    parts.append(INSTRUCTION[dataset])
    return "\n".join(parts)


def build_prompt(dataset: str, item: dict[str, Any], context_mode: str, retrieved: Sequence[str] | None = None,
                 context_tokens: int = 4096, max_new_tokens: int = 32,
                 count_tokens: Callable[[str], int] | None = None,
                 overhead_tokens: int = TEMPLATE_OVERHEAD_TOKENS) -> Prompt:
    """Build chat messages. context_mode: closed, full, gold, retrieved, or retention:<domain>."""
    count = count_tokens or heuristic_token_count
    budget = context_tokens - max_new_tokens - overhead_tokens
    budget -= count(SYSTEM[dataset]) + count(_user_text(dataset, item, None))
    units: list[str] = []
    truncated = False
    if context_mode == "retrieved":
        units, truncated = fit_units(list(retrieved or []), budget, count)
    elif context_mode == "full" and dataset == "finqa":
        units, truncated = fit_units([u for _, u in finqa_units(item)], budget, count)
    elif context_mode == "gold" and dataset == "pubmedqa":
        units, truncated = fit_units(list(item.get("contexts", [])), budget, count)
    elif context_mode not in ("closed", "full") and not context_mode.startswith("retention:"):
        raise ValueError(f"unsupported context_mode {context_mode!r} for {dataset}")
    if truncated:
        log.debug("context truncated for %s/%s", dataset, item.get("query_id"))
    context = "\n".join(units) if units else None
    messages = [{"role": "system", "content": SYSTEM[dataset]},
                {"role": "user", "content": _user_text(dataset, item, context)}]
    return Prompt(messages, truncated, units)


def render(messages: list[dict[str, str]], tokenizer: Any = None, add_generation_prompt: bool = True,
           template_kwargs: dict[str, Any] | None = None) -> str:
    """Apply the tokenizer's chat template when it has one, else a plain role-tagged fallback.

    ``template_kwargs`` (e.g. a fixed Llama 3 ``date_string``) must be identical at train and eval time.
    """
    kw = dict(template_kwargs or {})
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                 **kw)
        except Exception as exc:  # some templates reject a system role
            log.warning("chat template failed (%s); merging system into user", exc)
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            merged += messages[2:]
            return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                 **kw)
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def target_text(dataset: str, item: dict[str, Any]) -> str:
    """Training target: the bare answer the instruction asks for."""
    if dataset == "finqa":
        return str(item.get("gold_text") or item["gold"])
    return str(item["gold"])
