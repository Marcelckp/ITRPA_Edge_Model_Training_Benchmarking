"""Project wall-clock time for a plan on a device from pilot queries already in the results DB."""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from statistics import mean
from typing import Any, Sequence

from .config import ExpConfig

log = logging.getLogger(__name__)


def estimate(conn: sqlite3.Connection, configs: Sequence[ExpConfig], warmup: int = 3, idle_s: float = 60.0,
             startup_s: float = 20.0, cooldown_s: float = 0.0, repeats: int = 1) -> dict[str, Any]:
    """Per device: projected hours. Configs without pilot rows borrow the mean of the closest
    (model, precision, dataset) group, then the device mean."""
    rows = conn.execute("SELECT r.config_id, r.e2e_latency_s, c.model, c.precision, c.dataset, c.device "
                        "FROM runs r JOIN configs c USING(config_id) WHERE r.error IS NULL AND r.e2e_latency_s IS NOT NULL"
                        ).fetchall()
    by_cfg: dict[str, list[float]] = defaultdict(list)
    by_group: dict[tuple, list[float]] = defaultdict(list)
    by_dev: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(r["e2e_latency_s"])
        by_group[(r["device"], r["model"], r["precision"], r["dataset"])].append(r["e2e_latency_s"])
        by_dev[r["device"]].append(r["e2e_latency_s"])
    out: dict[str, Any] = {}
    for c in configs:
        d = out.setdefault(c.device, {"configs": 0, "queries": 0, "seconds": 0.0, "measured": 0, "imputed": 0,
                                      "missing": 0})
        if by_cfg.get(c.config_id):
            per_q, d["measured"] = mean(by_cfg[c.config_id]), d["measured"] + 1
        elif by_group.get((c.device, c.model, c.precision, c.dataset)):
            per_q, d["imputed"] = mean(by_group[(c.device, c.model, c.precision, c.dataset)]), d["imputed"] + 1
        elif by_dev.get(c.device):
            per_q, d["imputed"] = mean(by_dev[c.device]), d["imputed"] + 1
        else:
            d["missing"] += 1
            continue
        n = int(c.n_items or 0)
        blocks = max(1, repeats)
        d["configs"] += 1
        d["queries"] += n * blocks
        d["seconds"] += blocks * ((n + warmup) * per_q + idle_s + startup_s + cooldown_s)
    for d in out.values():
        d["hours"] = d["seconds"] / 3600.0
    log.info("estimate: %s", out)
    return out
