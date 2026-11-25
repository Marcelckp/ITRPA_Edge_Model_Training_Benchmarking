"""Answer parsing and scoring for PubMedQA, FinQA and ARC-Easy."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

PUBMEDQA_LABELS = ("yes", "no", "maybe")
FINQA_REL_TOL = 1e-2

_LABEL_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_NUM_RE = re.compile(
    r"(?P<neg>[-\u2212])?\s*(?P<open>\()?\s*(?P<cur>[$\u20ac\u00a3])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s*(?P<close>\))?\s*(?P<pct>%|percent)?",
    re.IGNORECASE,
)


@dataclass
class Score:
    parsed: str | None
    parse_ok: bool
    correct: bool


def parse_pubmedqa(text: str | None) -> str | None:
    if not text:
        return None
    m = _LABEL_RE.search(text)
    return m.group(1).lower() if m else None


def parse_yes_no_exact(text: str | None) -> str | None:
    """Exact yes/no answer: the whole reply, ignoring case, whitespace and trailing punctuation."""
    if not text:
        return None
    t = text.strip().strip(".!\"'").strip().lower()
    return t if t in ("yes", "no") else None


@dataclass
class ParsedNumber:
    value: float
    is_percent: bool
    decimals: int


def parse_number(text: str | None) -> ParsedNumber | None:
    """First number in the text; handles %, thousands commas, currency, unicode minus and (accounting) negatives."""
    if text is None:
        return None
    m = _NUM_RE.search(str(text).strip())
    if not m:
        return None
    raw = m.group("num").replace(",", "")
    value = float(raw)
    if m.group("neg") or (m.group("open") and m.group("close")):
        value = -value
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return ParsedNumber(value, bool(m.group("pct")), decimals)


def _decimals_of(x: float) -> int:
    s = repr(float(x))
    if "e" in s or "E" in s:
        return 6
    frac = s.split(".")[1] if "." in s else ""
    return 0 if frac == "0" else len(frac)


def numbers_match(pred: float, gold: float, rel_tol: float = FINQA_REL_TOL) -> bool:
    """Relative tolerance only; rounding the prediction to the gold's decimals would accept 2.6 for 3."""
    if math.isnan(pred) or math.isnan(gold):
        return False
    if gold == 0:
        return abs(pred) <= 1e-9
    return abs(pred - gold) <= rel_tol * abs(gold)


def finqa_correct(pred_text: str | None, gold: str | float, gold_text: str | None = None) -> Score:
    """Correct if the first number matches the gold within relative tolerance after normalisation.

    Percentages and fractions are interchangeable in both directions (12.5% vs 0.125), because FinQA stores
    some percentage answers as executed ratios.
    """
    gold_str = str(gold).strip().lower()
    if gold_str in ("yes", "no"):
        p = parse_yes_no_exact(pred_text)
        return Score(p, p is not None, p == gold_str)
    pn = parse_number(pred_text)
    if pn is None:
        return Score(None, False, False)
    gn = parse_number(gold_str) if not isinstance(gold, (int, float)) else ParsedNumber(float(gold), False, _decimals_of(gold))
    if gn is None:
        return Score(f"{pn.value:g}", True, False)
    golds = [gn]
    gt = parse_number(gold_text) if gold_text else None
    if gt is not None:
        golds.append(gt)
    # Scale by 100 only when exactly one side carries a % sign, so "1250%" never matches gold "12.5%".
    gold_pct = any(g.is_percent for g in golds)
    gold_values = {g.value for g in golds}
    if gold_pct and not pn.is_percent:
        gold_values |= {g.value / 100.0 for g in golds if g.is_percent}
    candidates = {pn.value}
    if pn.is_percent and not gold_pct:
        candidates.add(pn.value / 100.0)
    ok = any(numbers_match(c, g) for c in candidates for g in gold_values)
    parsed = f"{pn.value:g}%" if pn.is_percent else f"{pn.value:g}"
    return Score(parsed, True, ok)


# No bare-letter fallback: it would match the article "A" in replies such as "A plant needs light".
_ARC_PATTERNS = (
    re.compile(r"^\s*\(?([A-E])\)?\s*(?:[.:)\]]|$)"),
    re.compile(r"(?i:answer)\s*(?i:is)?\s*[:\-]?\s*\(?([A-E])(?![A-Za-z])"),
    re.compile(r"\(([A-E])\)"),
    re.compile(r"(?<![A-Za-z(])([A-E])\)"),
)


def parse_arc(text: str | None, valid: Sequence[str] = ("A", "B", "C", "D", "E")) -> str | None:
    if not text:
        return None
    for pat in _ARC_PATTERNS:
        for m in pat.finditer(text):
            letter = m.group(1).upper()
            if letter in valid:
                return letter
    return None


def score(dataset: str, raw_output: str | None, item: dict[str, Any]) -> Score:
    if dataset == "pubmedqa":
        p = parse_pubmedqa(raw_output)
        return Score(p, p is not None, p == item["gold"])
    if dataset == "finqa":
        return finqa_correct(raw_output, item["gold"], item.get("gold_text"))
    if dataset == "arc_easy":
        valid = [c["label"] for c in item.get("choices", [])] or list("ABCDE")
        p = parse_arc(raw_output, valid)
        return Score(p, p is not None, p == item["gold"])
    raise ValueError(f"unknown dataset {dataset}")


def accuracy(correct: Sequence[bool | int]) -> float | None:
    return sum(bool(c) for c in correct) / len(correct) if correct else None


def macro_f1(golds: Sequence[str], preds: Sequence[str | None], labels: Sequence[str] = PUBMEDQA_LABELS) -> float | None:
    """Macro-F1 over ``labels``; an unparsed prediction counts as a miss for its gold class."""
    if not golds:
        return None
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else None


def per_class_recall(golds: Sequence[str], preds: Sequence[str | None],
                     labels: Sequence[str] = PUBMEDQA_LABELS) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for lab in labels:
        n = sum(1 for g in golds if g == lab)
        out[lab] = (sum(1 for g, p in zip(golds, preds) if g == lab and p == lab) / n) if n else None
    return out
