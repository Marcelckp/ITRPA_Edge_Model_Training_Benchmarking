"""SQLite results database (DESIGN.md section 4)."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS environment(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS configs(config_id TEXT PRIMARY KEY, model TEXT, strategy TEXT, precision TEXT,
  engine TEXT, tier TEXT, device TEXT, dataset TEXT, context_mode TEXT, k INTEGER, model_path TEXT,
  seed INTEGER, temperature REAL, max_new_tokens INTEGER, is_mock INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS runs(run_id INTEGER PRIMARY KEY, config_id TEXT, query_id TEXT, block_id TEXT,
  started_at TEXT, prompt_tokens INTEGER, output_tokens INTEGER, ttft_s REAL, tpot_s REAL, e2e_latency_s REAL,
  tokens_per_s REAL, server_prompt_ms REAL, server_predicted_ms REAL, peak_mem_mb REAL, mem_kind TEXT,
  peak_vram_mb REAL, energy_j REAL, avg_power_w REAL, temp_c REAL, throttled INTEGER, truncated INTEGER,
  retrieval_hit INTEGER, raw_output TEXT, parsed_answer TEXT, parse_ok INTEGER, gold_answer TEXT,
  correct INTEGER, error TEXT, is_mock INTEGER NOT NULL DEFAULT 0, UNIQUE(config_id, query_id));
CREATE TABLE IF NOT EXISTS energy_blocks(block_id TEXT PRIMARY KEY, config_id TEXT, device TEXT,
  repeat_index INTEGER, started_at TEXT, ended_at TEXT, n_queries INTEGER, meter_start_mwh REAL,
  meter_end_mwh REAL, energy_j REAL, idle_power_w REAL, idle_energy_j REAL, source TEXT,
  is_mock INTEGER NOT NULL DEFAULT 0, idle_started_at TEXT, idle_meter_start_mwh REAL);
CREATE TABLE IF NOT EXISTS training_runs(train_id INTEGER PRIMARY KEY, model TEXT, strategy TEXT,
  dataset TEXT, seed INTEGER, learning_rate REAL, selected INTEGER, val_metric REAL, n_train_examples INTEGER,
  epochs REAL, peak_vram_mb REAL, duration_s REAL, energy_j REAL, avg_power_w REAL, trainable_params INTEGER,
  total_params INTEGER, final_loss REAL, optimizer TEXT, grad_checkpointing INTEGER,
  is_mock INTEGER NOT NULL DEFAULT 0, peak_mem_mb REAL, mem_kind TEXT, device TEXT, cpu_s REAL);
CREATE TABLE IF NOT EXISTS artifacts(artifact_id INTEGER PRIMARY KEY, model TEXT, strategy TEXT, dataset TEXT,
  precision TEXT, path TEXT, size_bytes INTEGER, sha256 TEXT, train_id INTEGER);
CREATE TABLE IF NOT EXISTS telemetry(sample_id INTEGER PRIMARY KEY, scope TEXT, ref_id TEXT, t REAL,
  power_w REAL, vram_mb REAL, mem_mb REAL, temp_c REAL);
CREATE TABLE IF NOT EXISTS cpu_usage(usage_id INTEGER PRIMARY KEY, scope TEXT, ref_id TEXT, config_id TEXT,
  device TEXT, started_at TEXT, ended_at TEXT, wall_s REAL, cpu_s REAL, peak_mem_mb REAL, mem_kind TEXT,
  n_queries INTEGER, source TEXT, coverage TEXT, is_mock INTEGER NOT NULL DEFAULT 0, UNIQUE(scope, ref_id, source));
CREATE TABLE IF NOT EXISTS energy_estimates(estimate_id INTEGER PRIMARY KEY, usage_id INTEGER, scope TEXT,
  ref_id TEXT, config_id TEXT, device TEXT, wall_s REAL, cpu_s REAL, mem_gb REAL, n_queries INTEGER,
  p_core_w_low REAL, p_core_w_central REAL, p_core_w_high REAL, p_mem_w_per_gb REAL, pue REAL,
  energy_j_low REAL, energy_j_central REAL, energy_j_high REAL, energy_basis TEXT, method_note TEXT,
  is_mock INTEGER NOT NULL DEFAULT 0, UNIQUE(usage_id, energy_basis));
"""

# Columns added after v2; connect() adds them to older databases.
MIGRATIONS = {"training_runs": (("peak_mem_mb", "REAL"), ("mem_kind", "TEXT"), ("device", "TEXT"), ("cpu_s", "REAL"))}

RUN_COLS = ("config_id", "query_id", "block_id", "started_at", "prompt_tokens", "output_tokens", "ttft_s",
            "tpot_s", "e2e_latency_s", "tokens_per_s", "server_prompt_ms", "server_predicted_ms", "peak_mem_mb",
            "mem_kind", "peak_vram_mb", "energy_j", "avg_power_w", "temp_c", "throttled", "truncated",
            "retrieval_hit", "raw_output", "parsed_answer", "parse_ok", "gold_answer", "correct", "error",
            "is_mock")
BLOCK_COLS = ("block_id", "config_id", "device", "repeat_index", "started_at", "ended_at", "n_queries",
              "meter_start_mwh", "meter_end_mwh", "energy_j", "idle_power_w", "idle_energy_j", "source",
              "is_mock", "idle_started_at", "idle_meter_start_mwh")
CONFIG_COLS = ("config_id", "model", "strategy", "precision", "engine", "tier", "device", "dataset",
               "context_mode", "k", "model_path", "seed", "temperature", "max_new_tokens", "is_mock")
TRAIN_COLS = ("model", "strategy", "dataset", "seed", "learning_rate", "selected", "val_metric", "n_train_examples", "epochs", "peak_vram_mb",
              "duration_s", "energy_j", "avg_power_w", "trainable_params", "total_params", "final_loss",
              "optimizer", "grad_checkpointing", "is_mock", "peak_mem_mb", "mem_kind", "device", "cpu_s")
ARTIFACT_COLS = ("model", "strategy", "dataset", "precision", "path", "size_bytes", "sha256", "train_id")
TELEMETRY_COLS = ("scope", "ref_id", "t", "power_w", "vram_mb", "mem_mb", "temp_c")
USAGE_COLS = ("scope", "ref_id", "config_id", "device", "started_at", "ended_at", "wall_s", "cpu_s", "peak_mem_mb",
              "mem_kind", "n_queries", "source", "coverage", "is_mock")
ESTIMATE_COLS = ("usage_id", "scope", "ref_id", "config_id", "device", "wall_s", "cpu_s", "mem_gb", "n_queries",
                 "p_core_w_low", "p_core_w_central", "p_core_w_high", "p_mem_w_per_gb", "pue", "energy_j_low",
                 "energy_j_central", "energy_j_high", "energy_basis", "method_note", "is_mock")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(path: str | os.PathLike) -> sqlite3.Connection:
    p = Path(path)
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL") if str(p) != ":memory:" else None
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, cols in MIGRATIONS.items():
        have = _columns(conn, table)
        for name, kind in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    conn.commit()


def set_env(conn: sqlite3.Connection, items: dict[str, Any]) -> None:
    conn.executemany("INSERT OR REPLACE INTO environment(key, value) VALUES(?, ?)",
                     [(k, None if v is None else str(v)) for k, v in items.items()])
    conn.commit()


def get_env(conn: sqlite3.Connection, prefix: str = "") -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM environment WHERE key LIKE ?", (prefix + "%",)).fetchall()
    return {r["key"]: r["value"] for r in rows}


def upsert_config(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    existing = conn.execute("SELECT * FROM configs WHERE config_id=?", (row["config_id"],)).fetchone()
    if existing is None:
        conn.execute(f"INSERT INTO configs({','.join(CONFIG_COLS)}) VALUES({','.join('?' * len(CONFIG_COLS))})",
                     [row.get(c) for c in CONFIG_COLS])
    elif existing["model_path"] != row.get("model_path"):
        conn.execute("UPDATE configs SET model_path=? WHERE config_id=?", (row.get("model_path"), row["config_id"]))
    conn.commit()


def done_query_ids(conn: sqlite3.Connection, config_id: str, include_errors: bool = True) -> set[str]:
    sql = "SELECT query_id FROM runs WHERE config_id=?"
    if not include_errors:
        sql += " AND error IS NULL"
    return {r[0] for r in conn.execute(sql, (config_id,))}


def config_is_mock(conn: sqlite3.Connection, config_id: str) -> int:
    r = conn.execute("SELECT is_mock FROM configs WHERE config_id=?", (config_id,)).fetchone()
    return int(r[0]) if r else 0


def insert_run(conn: sqlite3.Connection, row: dict[str, Any], replace: bool = False) -> bool:
    """Insert one query result. Returns False when the (config_id, query_id) pair already exists."""
    row = dict(row)
    row["is_mock"] = max(int(row.get("is_mock") or 0), config_is_mock(conn, row["config_id"]))
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    cur = conn.execute(f"{verb} INTO runs({','.join(RUN_COLS)}) VALUES({','.join('?' * len(RUN_COLS))})",
                       [row.get(c) for c in RUN_COLS])
    conn.commit()
    return cur.rowcount > 0


def upsert_energy_block(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert or update a block; fields absent from ``row`` keep their stored values."""
    existing = conn.execute("SELECT * FROM energy_blocks WHERE block_id=?", (row["block_id"],)).fetchone()
    merged = dict(existing) if existing else {}
    merged.update({k: v for k, v in row.items() if v is not None or k not in merged
                   or k in ("idle_power_w", "idle_energy_j")})
    merged["is_mock"] = int(merged.get("is_mock") or 0)
    if merged.get("config_id"):
        merged["is_mock"] = max(int(merged.get("is_mock") or 0), config_is_mock(conn, merged["config_id"]))
    if (merged.get("source") or "").startswith("usb_meter"):
        _meter_energy(merged)
    conn.execute(f"INSERT OR REPLACE INTO energy_blocks({','.join(BLOCK_COLS)}) "
                 f"VALUES({','.join('?' * len(BLOCK_COLS))})", [merged.get(c) for c in BLOCK_COLS])
    conn.commit()


def _iso_s(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _meter_energy(b: dict[str, Any]) -> None:
    """Gross, idle power and idle energy all from the same meter; idle is subtracted over exactly the
    interval between the two block readings."""
    i0, m0, m1 = b.get("idle_meter_start_mwh"), b.get("meter_start_mwh"), b.get("meter_end_mwh")
    if m0 is not None and m1 is not None:
        b["energy_j"] = (float(m1) - float(m0)) * 3.6
    if i0 is not None and m0 is not None and b.get("idle_started_at") and b.get("started_at"):
        dt = _iso_s(b["started_at"]) - _iso_s(b["idle_started_at"])
        b["idle_power_w"] = (float(m0) - float(i0)) * 3.6 / dt if dt > 0 else None
    if b.get("idle_power_w") is not None and b.get("started_at") and b.get("ended_at"):
        b["idle_energy_j"] = b["idle_power_w"] * (_iso_s(b["ended_at"]) - _iso_s(b["started_at"]))


def delete_block_runs(conn: sqlite3.Connection, block_id: str) -> int:
    cur = conn.execute("DELETE FROM runs WHERE block_id=?", (block_id,))
    conn.commit()
    return cur.rowcount


def insert_training_run(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    row = {**row, "is_mock": int(row.get("is_mock") or 0)}
    cur = conn.execute(f"INSERT INTO training_runs({','.join(TRAIN_COLS)}) VALUES({','.join('?' * len(TRAIN_COLS))})",
                       [row.get(c) for c in TRAIN_COLS])
    conn.commit()
    return int(cur.lastrowid)


def insert_artifact(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cur = conn.execute(f"INSERT INTO artifacts({','.join(ARTIFACT_COLS)}) VALUES({','.join('?' * len(ARTIFACT_COLS))})",
                       [row.get(c) for c in ARTIFACT_COLS])
    conn.commit()
    return int(cur.lastrowid)


def insert_telemetry(conn: sqlite3.Connection, scope: str, ref_id: str, samples: Iterable[dict[str, Any]]) -> int:
    rows = [(scope, ref_id, s.get("t"), s.get("power_w"), s.get("vram_mb"), s.get("mem_mb"), s.get("temp_c"))
            for s in samples]
    conn.executemany(f"INSERT INTO telemetry({','.join(TELEMETRY_COLS)}) VALUES(?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def upsert_cpu_usage(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Insert or replace one CPU-time window (unique per scope, ref_id and source); returns usage_id."""
    row = {**row, "is_mock": int(row.get("is_mock") or 0)}
    conn.execute(f"INSERT OR REPLACE INTO cpu_usage({','.join(USAGE_COLS)}) VALUES({','.join('?' * len(USAGE_COLS))})",
                 [row.get(c) for c in USAGE_COLS])
    conn.commit()
    return int(conn.execute("SELECT usage_id FROM cpu_usage WHERE scope=? AND ref_id=? AND source=?",
                            (row["scope"], row["ref_id"], row["source"])).fetchone()[0])


def upsert_energy_estimate(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    row = {**row, "is_mock": int(row.get("is_mock") or 0)}
    conn.execute(f"INSERT OR REPLACE INTO energy_estimates({','.join(ESTIMATE_COLS)}) "
                 f"VALUES({','.join('?' * len(ESTIMATE_COLS))})", [row.get(c) for c in ESTIMATE_COLS])
    conn.commit()


def has_mock(conn: sqlite3.Connection) -> bool:
    for table in ("configs", "runs", "training_runs", "energy_blocks"):
        if conn.execute(f"SELECT 1 FROM {table} WHERE is_mock=1 LIMIT 1").fetchone():
            return True
    return False


def merge_dbs(dest: str | os.PathLike, sources: Iterable[str | os.PathLike]) -> dict[str, int]:
    """Merge results DBs from several devices into ``dest``; re-merging the same source is a no-op."""
    conn = connect(dest)
    counts = {"configs": 0, "runs": 0, "energy_blocks": 0, "training_runs": 0, "artifacts": 0, "telemetry": 0, "environment": 0}
    for i, src in enumerate(sources):
        if Path(src).resolve() == Path(dest).resolve():
            continue
        alias = f"src{i}"
        conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(src),))
        try:
            conflicts = conn.execute(
                f"SELECT s.config_id FROM {alias}.configs s JOIN main.configs m USING(config_id) "
                "WHERE s.model IS NOT m.model OR s.strategy IS NOT m.strategy OR s.device IS NOT m.device "
                "OR s.is_mock IS NOT m.is_mock").fetchall()
            for c in conflicts:
                log.warning("config %s differs between %s and %s; keeping destination row", c[0], src, dest)
            before = conn.total_changes
            conn.execute(f"INSERT OR IGNORE INTO main.configs SELECT {','.join(CONFIG_COLS)} FROM {alias}.configs")
            counts["configs"] += conn.total_changes - before
            before = conn.total_changes
            cols = ",".join(RUN_COLS)
            conn.execute(f"INSERT OR IGNORE INTO main.runs({cols}) SELECT {cols} FROM {alias}.runs")
            counts["runs"] += conn.total_changes - before
            before = conn.total_changes
            cols = ",".join(BLOCK_COLS)
            conn.execute(f"INSERT OR IGNORE INTO main.energy_blocks({cols}) SELECT {cols} FROM {alias}.energy_blocks")
            counts["energy_blocks"] += conn.total_changes - before
            before = conn.total_changes
            src_cols = _columns(conn, "training_runs", alias)
            cols = ",".join(c for c in TRAIN_COLS if c in src_cols)
            conn.execute(
                f"INSERT INTO main.training_runs({cols}) SELECT {cols} FROM {alias}.training_runs s WHERE NOT EXISTS("
                "SELECT 1 FROM main.training_runs m WHERE m.model IS s.model AND m.strategy IS s.strategy AND "
                "m.seed IS s.seed AND m.dataset IS s.dataset AND m.learning_rate IS s.learning_rate AND "
                "m.duration_s IS s.duration_s AND "
                "m.final_loss IS s.final_loss AND m.is_mock IS s.is_mock)")
            counts["training_runs"] += conn.total_changes - before
            before = conn.total_changes
            cols = ",".join(ARTIFACT_COLS)
            conn.execute(
                f"INSERT INTO main.artifacts({cols}) SELECT {cols} FROM {alias}.artifacts s WHERE NOT EXISTS("
                "SELECT 1 FROM main.artifacts m WHERE m.path IS s.path AND m.sha256 IS s.sha256)")
            counts["artifacts"] += conn.total_changes - before
            before = conn.total_changes
            cols = ",".join(TELEMETRY_COLS)
            conn.execute(
                f"INSERT INTO main.telemetry({cols}) SELECT {cols} FROM {alias}.telemetry s WHERE NOT EXISTS("
                "SELECT 1 FROM main.telemetry m WHERE m.scope IS s.scope AND m.ref_id IS s.ref_id AND m.t IS s.t)")
            counts["telemetry"] += conn.total_changes - before
            before = conn.total_changes
            conn.execute(f"INSERT OR IGNORE INTO main.environment SELECT key, value FROM {alias}.environment")
            counts["environment"] += conn.total_changes - before
            conn.commit()
        finally:
            conn.execute(f"DETACH DATABASE {alias}")
    conn.close()
    log.info("merged into %s: %s", dest, counts)
    return counts
