"""Estimated energy from container CPU time (Green Algorithms model), for hosts with no energy counter.

Every figure produced here is an ESTIMATE from CPU time, not a measurement. It is stored in the
``energy_estimates`` table with ``energy_basis = 'estimated:green-algorithms'`` and never in ``energy_j`` columns,
so measured-energy analyses and the H1b energy decision never see it.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import db
from .config import DEFAULT_CONFIG_DIR, load_yaml
from .telemetry.cgroup import HostSample, iso_to_unix, read_host_csv, window_usage
from .telemetry.energy import J_PER_KWH

log = logging.getLogger(__name__)

DEFAULT_MODEL_FILE = DEFAULT_CONFIG_DIR / "energy-model.yaml"
ENERGY_BASIS = "estimated:green-algorithms"
LABEL = "estimated from CPU time, not metered (Green Algorithms model)"
LEVELS = ("low", "central", "high")


@dataclass
class EnergyModel:
    p_core_w: dict[str, float]
    p_mem_w_per_gb: float
    pue: float = 1.0
    threads_per_core: float = 1.0
    cpu_model: str = ""
    energy_basis: str = ENERGY_BASIS
    label: str = LABEL
    sources: str = ""

    def energy_j(self, cpu_s: float, wall_s: float, mem_gb: float | None, level: str = "central") -> float:
        """Green Algorithms: E = PUE x (core-seconds x P_core + wall-seconds x mem_GB x P_mem)."""
        core_s = cpu_s / self.threads_per_core
        mem_term = wall_s * (mem_gb or 0.0) * self.p_mem_w_per_gb
        return self.pue * (core_s * self.p_core_w[level] + mem_term)

    def note(self) -> str:
        pc = self.p_core_w
        return (f"{self.label}: E = PUE x (CPU-s x P_core + wall-s x mem_GB x P_mem); CPU {self.cpu_model}, "
                f"P_core {pc['low']:.3g}/{pc['central']:.3g}/{pc['high']:.3g} W (low/central/high), "
                f"P_mem {self.p_mem_w_per_gb:g} W/GB, PUE {self.pue:g}. {self.sources}").strip()


def load_model(path: str | Path = DEFAULT_MODEL_FILE) -> EnergyModel:
    d = load_yaml(path)
    pc = {k: float(d["p_core_w"][k]) for k in LEVELS}
    if not pc["low"] <= pc["central"] <= pc["high"]:
        raise ValueError("energy model: p_core_w must satisfy low <= central <= high")
    if d.get("energy_basis", ENERGY_BASIS) != ENERGY_BASIS or "not metered" not in d.get("label", LABEL):
        raise ValueError("energy model: label must say 'not metered' and basis must be estimated:green-algorithms")
    sources = " ".join(str(d[k]).strip() for k in ("p_core_source", "p_mem_source") if d.get(k))
    return EnergyModel(pc, float(d["p_mem_w_per_gb"]), float(d.get("pue", 1.0)), float(d.get("threads_per_core", 1.0)),
                       str(d.get("cpu_model", "")), ENERGY_BASIS, str(d.get("label", LABEL)), sources)


def _run_windows(conn: sqlite3.Connection, device: str) -> list[tuple[dict[str, Any], float, float]]:
    rows = conn.execute(
        "SELECT r.config_id, r.query_id, r.started_at, r.e2e_latency_s, r.is_mock FROM runs r JOIN configs c "
        "USING(config_id) WHERE c.device=? AND r.error IS NULL AND r.e2e_latency_s IS NOT NULL", (device,)).fetchall()
    out = []
    for r in rows:
        t0 = iso_to_unix(r["started_at"])
        out.append(({"scope": "query", "ref_id": f"{r['config_id']}:{r['query_id']}", "config_id": r["config_id"],
                     "n_queries": 1, "is_mock": r["is_mock"], "started_at": r["started_at"]},
                    t0, t0 + float(r["e2e_latency_s"])))
    return out


def _block_windows(conn: sqlite3.Connection, device: str) -> list[tuple[dict[str, Any], float, float]]:
    rows = conn.execute("SELECT * FROM energy_blocks WHERE device=? AND ended_at IS NOT NULL", (device,)).fetchall()
    return [({"scope": "block", "ref_id": b["block_id"], "config_id": b["config_id"], "n_queries": b["n_queries"],
              "is_mock": b["is_mock"], "started_at": b["started_at"], "ended_at": b["ended_at"]},
             iso_to_unix(b["started_at"]), iso_to_unix(b["ended_at"])) for b in rows]


def import_host_csv(conn: sqlite3.Connection, csv_path: str | Path, device: str,
                    source: str = "host-cgroup-csv") -> dict[str, int]:
    """Join 1 Hz host cgroup samples to every query and block window of ``device``; writes cpu_usage rows."""
    samples: list[HostSample] = read_host_csv(csv_path)
    counts = {"query": 0, "block": 0, "uncovered": 0}
    for meta, t0, t1 in _block_windows(conn, device) + _run_windows(conn, device):
        u = window_usage(samples, t0, t1)
        if u.cpu_s is None:
            counts["uncovered"] += 1
            continue
        db.upsert_cpu_usage(conn, {**meta, "device": device, "wall_s": u.wall_s, "cpu_s": u.cpu_s,
                                   "peak_mem_mb": u.peak_mem_mb, "mem_kind": "cgroup-usage (incl. page cache)",
                                   "source": source, "coverage": f"{u.coverage}, {u.n_samples} samples"})
        counts[meta["scope"]] += 1
    log.info("joined %s to %s windows: %s", csv_path, device, counts)
    return counts


def compute(conn: sqlite3.Connection, model: EnergyModel, scopes: Iterable[str] = ("query", "block", "train"),
            mem_gb_override: float | None = None) -> int:
    """Write one energy_estimates row (low/central/high) per cpu_usage row; returns the count."""
    scopes = tuple(scopes)
    q = f"SELECT * FROM cpu_usage WHERE scope IN ({','.join('?' * len(scopes))}) AND cpu_s IS NOT NULL"
    n = 0
    for u in conn.execute(q, scopes).fetchall():
        mem_gb = mem_gb_override if mem_gb_override is not None else (
            u["peak_mem_mb"] / 1024 if u["peak_mem_mb"] is not None else None)
        row = {"usage_id": u["usage_id"], "scope": u["scope"], "ref_id": u["ref_id"], "config_id": u["config_id"],
               "device": u["device"], "wall_s": u["wall_s"], "cpu_s": u["cpu_s"], "mem_gb": mem_gb,
               "n_queries": u["n_queries"], "p_mem_w_per_gb": model.p_mem_w_per_gb, "pue": model.pue,
               "energy_basis": model.energy_basis, "method_note": model.note(), "is_mock": u["is_mock"]}
        for lv in LEVELS:
            row[f"p_core_w_{lv}"] = model.p_core_w[lv]
            row[f"energy_j_{lv}"] = model.energy_j(u["cpu_s"], u["wall_s"], mem_gb, lv)
        db.upsert_energy_estimate(conn, row)
        n += 1
    return n


def _money(df: pd.DataFrame, tariff: dict[str, Any], per: float, prefix: str) -> pd.DataFrame:
    t, g = tariff["tariff_zar_per_kwh"], tariff["grid_kgco2e_per_kwh"]
    for lv in LEVELS:
        kwh = df[f"j_{prefix}_{lv}"] * per / J_PER_KWH
        df[f"zar_{lv}"] = kwh * t
        df[f"kgco2e_{lv}"] = kwh * g
    return df


def summary_tables(conn: sqlite3.Connection, tariff: dict[str, Any], allow_mock: bool = False) -> dict[str, pd.DataFrame]:
    """Block-level J/query per config (ZAR and CO2e per 1000 queries) and per-run training estimates."""
    mock = "" if allow_mock else " AND e.is_mock=0"
    out: dict[str, pd.DataFrame] = {}
    b = pd.read_sql_query(
        "SELECT e.*, c.model, c.strategy, c.precision, c.dataset, c.k, c.context_mode FROM energy_estimates e "
        f"JOIN configs c USING(config_id) WHERE e.scope='block'{mock}", conn)
    if not b.empty:
        g = b.groupby(["device", "model", "strategy", "precision", "dataset", "k", "context_mode"], dropna=False)
        s = g.agg(n_queries=("n_queries", "sum"), wall_s=("wall_s", "sum"), cpu_s=("cpu_s", "sum"),
                  mem_gb=("mem_gb", "max"), **{f"e_{lv}": (f"energy_j_{lv}", "sum") for lv in LEVELS}).reset_index()
        s["mean_cores_busy"] = s.cpu_s / s.wall_s
        for lv in LEVELS:
            s[f"j_per_query_{lv}"] = s[f"e_{lv}"] / s.n_queries
        s = _money(s, tariff, 1000.0, "per_query")
        s = s.drop(columns=[f"e_{lv}" for lv in LEVELS])
        s.columns = [c if not c.startswith(("zar_", "kgco2e_")) else c.replace("_", "_per_1000_queries_", 1)
                     for c in s.columns]
        s["energy_basis"], s["label"] = ENERGY_BASIS, LABEL
        out["estimated_energy_inference"] = s
    t = pd.read_sql_query(
        "SELECT e.*, t.model, t.strategy, t.dataset, t.learning_rate, t.seed, t.selected, t.duration_s "
        "FROM energy_estimates e JOIN training_runs t ON t.train_id = CAST(e.ref_id AS INTEGER) "
        f"WHERE e.scope='train'{mock}", conn)
    if not t.empty:
        for lv in LEVELS:
            t[f"j_run_{lv}"] = t[f"energy_j_{lv}"]
        t = _money(t, tariff, 1.0, "run")
        t["kwh_central"] = t.energy_j_central / J_PER_KWH
        t["energy_basis"], t["label"] = ENERGY_BASIS, LABEL
        out["estimated_energy_training"] = t[["model", "strategy", "dataset", "learning_rate", "seed", "selected",
                                              "duration_s", "cpu_s", "mem_gb", "energy_j_low", "energy_j_central",
                                              "energy_j_high", "kwh_central", "zar_low", "zar_central", "zar_high",
                                              "kgco2e_central", "energy_basis", "label"]]
    return out


def write_outputs(tables: dict[str, pd.DataFrame], model: EnergyModel, tariff: dict[str, Any], out_dir: str | Path,
                  caption: str | None = None) -> list[str]:
    from .report import to_markdown

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    head = (f"> **Estimated energy: {LABEL}.** These figures are not measurements. {model.note()} "
            f"Tariff R{tariff['tariff_zar_per_kwh']:g}/kWh ({tariff['tariff_source']}); grid factor "
            f"{tariff['grid_kgco2e_per_kwh']:g} kg CO2e/kWh ({tariff['grid_source']}).")
    parts = [f"# Estimated energy ({LABEL})", "", head, ""]
    if caption:
        parts += [f"_{caption}_", ""]
    titles = {"estimated_energy_inference": "Inference, block level, per config (estimated, not metered)",
              "estimated_energy_training": "Adaptation runs (estimated, not metered)"}
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        written.append(str(p))
        parts += [f"## {titles.get(name, name)}", "", to_markdown(df.drop(columns=["label"], errors="ignore")), ""]
    if not tables:
        parts += ["No CPU-time windows were found, so nothing was estimated.", ""]
    md = out / "estimated_energy.md"
    md.write_text("\n".join(parts))
    written.append(str(md))
    return written
