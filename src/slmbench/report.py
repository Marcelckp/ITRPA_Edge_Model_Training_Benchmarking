"""Dissertation artefacts: Markdown tables, PNG figures and the hypothesis-decision table."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import MOCK_WATERMARK, db
from .analysis import _settings, analyze

log = logging.getLogger(__name__)


class ReportRefused(RuntimeError):
    pass


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.3e}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def to_markdown(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    cols = [c for c in (columns or df.columns) if c in df.columns]
    if not cols or set(cols) == {"watermark"}:
        return "_No data._\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _write_md(path: Path, title: str, body: str, mock: bool, note: str | None = None) -> None:
    mark = f"**{MOCK_WATERMARK}**\n\n" if mock else ""
    parts = [mark, f"## {title}\n\n"]
    if note:
        parts.append(f"{note}\n\n")
    parts += [body, f"\n{mark}" if mock else ""]
    path.write_text("".join(parts), encoding="utf-8")


def _figure(path: Path, mock: bool, draw, caption: str | None = None) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ok = draw(ax)
    if not ok:
        plt.close(fig)
        return False
    if mock:
        fig.text(0.5, 0.5, MOCK_WATERMARK, fontsize=26, color="red", alpha=0.35, ha="center", va="center",
                 rotation=25, transform=fig.transFigure)
    fig.tight_layout()
    if caption:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.22))
        fig.text(0.01, 0.01, caption, fontsize=6, ha="left", va="bottom", wrap=True)
    meta = {"Description": MOCK_WATERMARK if mock else (caption or "slmbench report"), "Software": "slmbench"}
    fig.savefig(path, dpi=150, metadata=meta)
    plt.close(fig)
    return True


def _bar_accuracy(cs: pd.DataFrame, dataset: str, plan: str, precisions: Sequence[str]):
    """Grouped bars, one bar per precision with that config's own bootstrap CI (bounds are never averaged)."""
    def draw(ax) -> bool:
        d = cs[(cs.plan == plan) & (cs.dataset == dataset)]
        if d.empty:
            return False
        groups = sorted({(m, v) for m, v in zip(d.model, d.variant)})
        precs = [p for p in precisions if p in set(d.precision)]
        width = 0.8 / max(1, len(precs))
        for j, prec in enumerate(precs):
            sub = d[d.precision == prec].set_index(["model", "variant"])
            xs, ys, lo, hi = [], [], [], []
            for i, g in enumerate(groups):
                if g in sub.index:
                    r = sub.loc[g]
                    xs.append(i + (j - (len(precs) - 1) / 2) * width)
                    ys.append(r.accuracy)
                    lo.append(r.accuracy - r.acc_lo)
                    hi.append(r.acc_hi - r.accuracy)
            ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, label=prec)
        ax.set_xticks(range(len(groups)), [f"{m}\n{v}" for m, v in groups], rotation=60, fontsize=7)
        ax.set_ylabel("accuracy (95% bootstrap CI)")
        ax.set_title(f"{dataset}: accuracy of record by model, strategy and precision")
        ax.legend(fontsize=7)
        return True
    return draw


def _line_latency_k(cs: pd.DataFrame):
    def draw(ax) -> bool:
        d = cs[(cs.tier == "edge") & (cs.strategy == "rag")]
        if d.empty:
            return False
        for keys, g in d.groupby(["device", "model", "dataset", "precision"]):
            g = g.sort_values("k")
            ax.plot(g.k, g.ttft_primary_s_median, marker="o", label="/".join(map(str, keys)))
        ax.set_xlabel("K retrieved passages")
        ax.set_ylabel("median TTFT (s)")
        ax.set_title("Edge TTFT vs K")
        ax.legend(fontsize=5, ncol=2)
        return True
    return draw


def _box_memory(cs: pd.DataFrame):
    def draw(ax) -> bool:
        d = cs[cs.tier == "edge"]
        if d.empty or d.peak_mem_mb_median.isna().all():
            return False
        g = d.groupby(["device", "precision"]).peak_mem_mb_median.mean().unstack()
        g.plot.bar(ax=ax)
        ax.set_ylabel("peak server memory (MB, median of configs)")
        ax.set_title("Edge memory by precision")
        return True
    return draw


def _bar_training(ts: pd.DataFrame):
    def draw(ax) -> bool:
        if ts is None or ts.empty or "peak_vram_mb_mean" not in ts:
            return False
        g = ts.groupby(["model", "strategy"]).peak_vram_mb_mean.mean().unstack()
        g.plot.bar(ax=ax)
        ax.set_ylabel("peak training memory (MB; VRAM on GPU, RSS on CPU)")
        ax.set_title("Adaptation memory by strategy")
        return True
    return draw


def _bar_energy(eb: pd.DataFrame):
    def draw(ax) -> bool:
        if eb is None or eb.empty or "net_j_per_query" not in eb:
            return False
        g = eb.groupby(["device", "precision"]).net_j_per_query.mean().unstack()
        g.plot.bar(ax=ax)
        ax.set_ylabel("net energy per query (J, block level)")
        ax.set_title("Energy per query by precision")
        return True
    return draw


def _bar_estimated_energy(est: pd.DataFrame | None):
    def draw(ax) -> bool:
        if est is None or est.empty:
            return False
        g = est.groupby(["device", "model", "precision"])[["j_per_query_low", "j_per_query_central",
                                                            "j_per_query_high"]].mean()
        c = g.j_per_query_central.unstack()
        err_lo = (g.j_per_query_central - g.j_per_query_low).unstack()
        err_hi = (g.j_per_query_high - g.j_per_query_central).unstack()
        c.plot.bar(ax=ax, yerr=[[err_lo[col].values, err_hi[col].values] for col in c.columns], capsize=2)
        ax.set_ylabel("ESTIMATED J per query (from CPU time, not metered)")
        ax.set_title("Estimated energy per query by precision (not metered)")
        return True
    return draw


DECISION_COLS = ["hypothesis", "family", "group", "comparison", "test", "n", "statistic", "p", "p_holm",
                 "effect", "ci", "decision", "reason"]

RULES = """Decision rules (DESIGN.md section 2; alpha = 0.05; decided per model, never pooled):

- H1a: descriptive; supported if peak VRAM ordering QLoRA < LoRA < FFT holds in all models (seed CV in training table).
- H1b memory: Friedman across F16/Q8_0/Q4_K_M + post-hoc Wilcoxon/Holm; energy: block bootstrap CIs ordered and non-overlapping (>= 3 blocks).
- H2a: non-inferiority, supported if the lower 95% Newcombe (method 10) bound of (candidate - FFT) > -delta.
- H2b: one-sided exact McNemar (LoRA > FFT, QLoRA > FFT), Holm within family.
- H3: Page's L across K = 3, 5, 10; diminishing returns if McNemar(K10 vs K5) is non-significant while Wilcoxon e2e(K10 > K5) is significant, or the bootstrap CI of [(Acc10 - Acc5)/5 - (Acc5 - Acc3)/2] lies below 0.
- Directional hypotheses (H1b, H2b, H3) are supported only when the Holm-adjusted p < alpha AND the effect is in the predicted direction.
- Floor rule: < 10 discordant pairs or every config < 5% accuracy => "not testable"; for H2a only the 5% accuracy condition applies.
- Holm family = hypothesis x dataset x device (column `family`). Errored queries removed listwise per family.
"""


def build_report(conn: sqlite3.Connection, matrix: dict, tariff: dict, out_dir: str | Path,
                 allow_mock: bool = False) -> dict[str, Any]:
    mock = db.has_mock(conn)
    if mock and not allow_mock:
        raise ReportRefused("results DB contains is_mock rows; refusing to write dissertation artefacts "
                            "(use --allow-mock for a watermarked test report)")
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    res = analyze(conn, matrix, tariff, out / "csv", allow_mock=allow_mock)
    s = _settings(matrix)
    ref_plans = [n for n, p in matrix["plans"].items() if p.get("reference")]
    tariff_note = (f"Tariff {tariff['tariff_zar_per_kwh']} ZAR/kWh ({tariff['tariff_source']}); grid factor "
                   f"{tariff['grid_kgco2e_per_kwh']} kgCO2e/kWh ({tariff['grid_source']}).")
    rep = matrix.get("report") or {}
    caption = rep.get("caption_note")
    unmeasured = rep.get("energy_not_measured")
    energy_tables = {"energy_blocks", "cost", "break_even"}
    written: list[str] = []

    def table(name: str, title: str, frame: pd.DataFrame | None, cols: Sequence[str] | None = None,
              note: str | None = None) -> None:
        p = out / "tables" / f"{name}.md"
        notes = [caption, note, f"Energy not measured: {unmeasured}" if unmeasured and name in energy_tables else None]
        _write_md(p, title, to_markdown(frame if frame is not None else pd.DataFrame(), cols), mock,
                  "\n\n".join(n for n in notes if n) or None)
        written.append(str(p))

    cs = res.get("config_summary", pd.DataFrame())
    table("accuracy", "Accuracy of record", cs[cs.plan.isin([s["acc_plan"], s["ret_plan"]])] if not cs.empty else cs,
          ["model", "variant", "precision", "dataset", "context_mode", "n", "n_errors", "accuracy", "acc_lo", "acc_hi",
           "macro_f1", "recall_yes", "recall_no", "recall_maybe", "parse_fail_rate", "truncated", "retrieval_hit_rate"])
    table("reference_arms", "Reference arms (gold context, vLLM bf16)",
          cs[cs.plan.isin(ref_plans)] if not cs.empty else cs,
          ["model", "variant", "precision", "engine", "dataset", "context_mode", "n", "accuracy", "acc_lo", "acc_hi"])
    table("edge_latency_memory", "Edge latency and memory", cs[cs.tier == "edge"] if not cs.empty else cs,
          ["device", "model", "variant", "precision", "dataset", "n", "ttft_primary_s_median", "ttft_primary_s_iqr",
           "tpot_primary_s_median", "e2e_latency_s_median", "e2e_latency_s_iqr", "peak_mem_mb_median", "mem_kind",
           "throttled", "parse_fail_rate"])
    table("energy_blocks", "Block-level energy per query", res.get("energy_blocks_summary"))
    table("training", "Adaptation cost (selected learning rate and cost-variability seeds)", res.get("training_summary"))
    table("cost", "Operating cost and emissions", res.get("cost"),
          ["device", "model", "variant", "precision", "dataset", "j_per_query", "energy_basis", "zar_per_1000_queries",
           "kgco2e_per_1000_queries", "zar_per_correct"], note=tariff_note)
    table("break_even", "Break-even query volume: fine-tuning vs RAG", res.get("break_even"), note=tariff_note)
    table("accuracy_parity", "Edge accuracy parity vs accuracy of record", res.get("accuracy_parity"))
    table("recall_at_k", "Retrieval recall@K", res.get("recall_at_k"))
    table("posthoc", "Post-hoc pairwise tests", res.get("posthoc"),
          ["family", "group", "metric", "level_a", "level_b", "test", "n", "statistic", "p", "p_holm", "r_rb", "odds_ratio",
           "not_testable"])

    from .energy_estimate import LABEL as EST_LABEL, summary_tables

    est = summary_tables(conn, tariff, allow_mock=allow_mock)
    est_note = (f"ESTIMATED energy ({EST_LABEL}); not a measurement and not used for any hypothesis decision. "
                f"Method and constants: TRAINING-CPU.md section 5. {tariff_note}")
    if "estimated_energy_inference" in est:
        table("estimated_energy_inference", "Estimated energy per query (estimated from CPU time, not metered)",
              est["estimated_energy_inference"],
              ["device", "model", "strategy", "precision", "dataset", "k", "n_queries", "mean_cores_busy", "mem_gb",
               "j_per_query_low", "j_per_query_central", "j_per_query_high", "zar_per_1000_queries_central",
               "kgco2e_per_1000_queries_central", "energy_basis"], note=est_note)
    if "estimated_energy_training" in est:
        table("estimated_energy_training", "Estimated adaptation energy (estimated from CPU time, not metered)",
              est["estimated_energy_training"], [c for c in est["estimated_energy_training"].columns if c != "label"],
              note=est_note)

    hp = out / "hypothesis_decisions.md"
    _write_md(hp, "Hypothesis decisions", to_markdown(res["hypothesis_tests"], DECISION_COLS), mock,
              "\n\n".join(n for n in (caption, RULES) if n))
    written.append(str(hp))

    figs = {
        "accuracy_pubmedqa.png": _bar_accuracy(cs, "pubmedqa", s["acc_plan"], s["precisions"]) if not cs.empty else None,
        "accuracy_finqa.png": _bar_accuracy(cs, "finqa", s["acc_plan"], s["precisions"]) if not cs.empty else None,
        "edge_ttft_vs_k.png": _line_latency_k(cs) if not cs.empty else None,
        "edge_memory_by_precision.png": _box_memory(cs) if not cs.empty else None,
        "training_vram.png": _bar_training(res.get("training_summary")),
        "energy_by_precision.png": _bar_energy(res.get("energy_blocks_summary")),
        "estimated_energy_by_precision.png": _bar_estimated_energy(est.get("estimated_energy_inference")),
    }
    for name, draw in figs.items():
        cap = caption
        if name.startswith("estimated_"):
            cap = " ".join(x for x in (caption, f"Energy {EST_LABEL}; bars = central, whiskers = low/high P_core.") if x)
        if draw is not None and _figure(out / "figures" / name, mock, draw, cap):
            written.append(str(out / "figures" / name))
        else:
            log.info("figure %s skipped: no data", name)
    manifest = {"generated_at": db.utcnow(), "is_mock": mock, "watermark": MOCK_WATERMARK if mock else None,
                "caption_note": caption, "tariff": tariff, "files": written}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
