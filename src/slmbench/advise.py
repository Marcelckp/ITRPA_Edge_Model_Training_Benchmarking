"""Decision-support advisor: rank feasible configurations under deployment constraints; static HTML output."""

from __future__ import annotations

import html
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import MOCK_WATERMARK, db
from . import stats as S
from .analysis import MockDataRefused, _settings, accuracy_parity, adaptation_energy, load_frames, metered_source

log = logging.getLogger(__name__)

MATCH_KEYS = ["model", "strategy", "precision", "dataset", "context_mode", "k"]


@dataclass
class Constraints:
    device: str
    ram_mb: float | None = None
    latency_budget_s: float | None = None
    monthly_queries: int = 10000
    min_accuracy: float | None = None
    dataset: str | None = None
    tariff_zar_per_kwh: float = 0.0
    grid_kgco2e_per_kwh: float = 0.0
    amortise_months: int = 12
    objective: str = "cost"  # cost | accuracy | latency


def candidate_table(conn: sqlite3.Connection, matrix: dict, c: Constraints, allow_mock: bool = False) -> pd.DataFrame:
    s = _settings(matrix)
    fr = load_frames(conn, matrix, allow_mock=allow_mock)
    df, blocks, training = fr["runs"], fr["blocks"], fr["training"]
    if df.empty:
        return pd.DataFrame()
    dev = df[(df.device == c.device) & df.error.isna()]
    if c.dataset:
        dev = dev[dev.dataset == c.dataset]
    ref = df[(df.plan == s["acc_plan"]) & df.error.isna()]
    adapt = adaptation_energy(training, s)
    parity = accuracy_parity(df, s)
    flagged = set(parity.loc[parity.flag_rerun_500, "config_id"]) if not parity.empty else set()
    rows = []
    for cid, g in dev.groupby("config_id"):
        f = g.iloc[0]
        sel = ref
        for k in MATCH_KEYS:
            sel = sel[sel[k] == f[k]]
        if len(sel):
            acc, acc_src = float(sel.correct.mean()), f"accuracy of record (n={len(sel)})"
        else:
            acc, acc_src = float(g.correct.mean()), f"this device (n={len(g)})"
        allb = blocks[blocks.config_id == cid] if not blocks.empty else blocks
        b = allb[allb.energy_j.notna() & allb.source.map(metered_source)] if len(allb) else allb
        if len(b):
            jq, basis = float(b.net_j_per_query.mean()), "block-level net"
        elif len(allb) and not allb.source.map(metered_source).any():
            jq, basis = None, "no metered energy"
        elif g.net_energy_j.notna().any():
            jq, basis = float(g.net_energy_j.mean()), "per-query net"
        elif g.energy_j.notna().any():
            jq, basis = float(g.energy_j.mean()), "per-query gross"
        else:
            jq, basis = None, "no energy data"
        a = adapt.get((f.model, f.strategy, f.dataset)) if f.strategy in ("fft", "lora", "qlora") else 0.0
        a = None if a is None or pd.isna(a) else float(a)
        monthly = None
        if jq is not None:
            monthly = S.cost_zar(jq * c.monthly_queries, c.tariff_zar_per_kwh)
            if a:
                monthly += S.cost_zar(a, c.tariff_zar_per_kwh) / max(1, c.amortise_months)
        parity_flag = cid in flagged and len(g) < 500
        rows.append({"config_id": cid, "parity_flag": parity_flag, "model": f.model, "strategy": f.strategy, "variant": f.variant,
                     "precision": f.precision, "dataset": f.dataset, "k": int(f.k),
                     "accuracy": acc, "accuracy_source": acc_src,
                     "p95_e2e_s": float(np.nanpercentile(g.e2e_latency_s, 95)),
                     "median_ttft_s": float(np.nanmedian(g.ttft_primary_s)),
                     "peak_mem_mb": float(g.peak_mem_mb.max()) if g.peak_mem_mb.notna().any() else None,
                     "j_per_query": jq, "energy_basis": basis, "adaptation_j": a,
                     "monthly_zar": monthly,
                     "monthly_kgco2e": S.co2e_kg(jq * c.monthly_queries, c.grid_kgco2e_per_kwh) if jq is not None else None,
                     "is_mock": int(f.get("is_mock", 0) or 0)})
    return pd.DataFrame(rows)


def rank(cands: pd.DataFrame, c: Constraints) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cands.empty:
        return cands, cands
    reasons = []
    for _, r in cands.iterrows():
        why = []
        if c.ram_mb is not None:
            if r.peak_mem_mb is None or pd.isna(r.peak_mem_mb):
                why.append("memory unknown")
            elif r.peak_mem_mb > c.ram_mb:
                why.append(f"memory {r.peak_mem_mb:.0f} MB > {c.ram_mb:.0f} MB")
        if c.latency_budget_s is not None and r.p95_e2e_s > c.latency_budget_s:
            why.append(f"p95 latency {r.p95_e2e_s:.2f} s > {c.latency_budget_s:.2f} s")
        if r.get("parity_flag"):
            why.append("edge answers disagree with accuracy of record (< parity threshold); re-run on 500 items")
        if c.min_accuracy is not None and r.accuracy < c.min_accuracy:
            why.append(f"accuracy {r.accuracy:.3f} < {c.min_accuracy:.3f}")
        reasons.append("; ".join(why))
    cands = cands.assign(infeasible_reason=reasons)
    feas = cands[cands.infeasible_reason == ""].copy()
    keys = {"cost": (["monthly_zar", "accuracy", "p95_e2e_s"], [True, False, True]),
            "accuracy": (["accuracy", "monthly_zar", "p95_e2e_s"], [False, True, True]),
            "latency": (["p95_e2e_s", "accuracy", "monthly_zar"], [True, False, True])}[c.objective]
    feas = feas.sort_values(keys[0], ascending=keys[1], na_position="last").reset_index(drop=True)
    feas.insert(0, "rank", range(1, len(feas) + 1))
    return feas, cands[cands.infeasible_reason != ""].reset_index(drop=True)


def _html_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "<p><em>None.</em></p>"

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return html.escape(f"{v:.4g}" if isinstance(v, float) else str(v))

    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{cell(r[c])}</td>" for c in cols) + "</tr>" for _, r in df.iterrows())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(feasible: pd.DataFrame, infeasible: pd.DataFrame, c: Constraints, mock: bool,
                tariff_source: str = "") -> str:
    cols = ["rank", "model", "variant", "precision", "dataset", "accuracy", "accuracy_source", "p95_e2e_s",
            "median_ttft_s", "peak_mem_mb", "j_per_query", "energy_basis", "monthly_zar", "monthly_kgco2e"]
    banner = f'<div class="mock">{html.escape(MOCK_WATERMARK)}</div>' if mock else ""
    cons = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(str(v))}</li>" for k, v in asdict(c).items())
    top = ""
    if not feasible.empty:
        r = feasible.iloc[0]
        top = (f"<p class='rec'>Recommended: <b>{html.escape(str(r.model))} / {html.escape(str(r.variant))} / "
               f"{html.escape(str(r.precision))}</b> (objective: {html.escape(c.objective)}).</p>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>slmbench advisor - {html.escape(c.device)}</title>
<style>body{{font-family:sans-serif;margin:2em;}}table{{border-collapse:collapse;font-size:12px;}}
td,th{{border:1px solid #999;padding:3px 6px;}}th{{background:#eee;}}
.mock{{background:#c00;color:#fff;font-size:20px;font-weight:bold;padding:8px;text-align:center;}}
.rec{{font-size:16px;}}</style></head><body>
{banner}
<h1>Deployment advisor: {html.escape(c.device)}</h1>
<h2>Constraints</h2><ul>{cons}<li><b>tariff source</b>: {html.escape(tariff_source)}</li></ul>
{top}
<h2>Feasible configurations</h2>{_html_table(feasible, cols)}
<h2>Excluded configurations</h2>{_html_table(infeasible, ["model", "variant", "precision", "dataset", "infeasible_reason"])}
<p>Monthly cost = per-query energy x monthly volume x tariff + adaptation energy x tariff / amortisation months.
Accuracy comes from the accuracy-of-record run on the identical GGUF file when available.</p>
{banner}
</body></html>
"""


def advise(conn: sqlite3.Connection, matrix: dict, c: Constraints, out_html: str | Path,
           tariff_source: str = "", allow_mock: bool = False) -> dict[str, Any]:
    mock = db.has_mock(conn)
    if mock and not allow_mock:
        raise MockDataRefused("results DB contains is_mock rows; refusing to advise (use --allow-mock)")
    cands = candidate_table(conn, matrix, c, allow_mock=allow_mock)
    feasible, infeasible = rank(cands, c)
    p = Path(out_html)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(feasible, infeasible, c, mock, tariff_source), encoding="utf-8")
    log.info("advisor report written to %s (%d feasible, %d excluded)", p, len(feasible), len(infeasible))
    return {"feasible": feasible, "infeasible": infeasible, "html": str(p), "is_mock": mock}
