"""Execute configuration blocks: randomised order, warm-up, idle baseline, cool-down gate, resumable inserts."""

from __future__ import annotations

import logging
import random
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import db
from .backends.base import Backend
from .config import ExpConfig
from .datasets import edge_subset, load_split
from .prompts import TEMPLATE_OVERHEAD_TOKENS, Prompt, build_prompt, heuristic_token_count
from .rag import load_rankings, top_k
from .scoring import score
from .telemetry.base import TelemetrySession
from .telemetry.energy import integrate_trapezoid
from .telemetry.pi_pmic import is_throttled_now, read_throttled

log = logging.getLogger(__name__)


@dataclass
class RunOptions:
    data_dir: str | Path = "data"
    warmup: int = 3
    idle_s: float = 60.0
    repeats: int = 3
    block_order_seed: int = 42
    cooldown_below_c: float | None = 50.0
    cooldown_max_wait_s: float = 900.0
    cooldown_poll_s: float = 5.0
    rerun_throttled: bool = True
    max_block_attempts: int = 2
    limit: int | None = None
    retry_errors: bool = False
    context_tokens: int = 4096
    overhead_tokens: int = TEMPLATE_OVERHEAD_TOKENS
    keep_raw_telemetry: bool = False
    block_energy: str | None = None  # usb_meter_manual | usb_meter_csv | None (use the power source)
    meter_prompt: Callable[[str], float] | None = None
    clock: Callable[[], float] = time.time
    throttle_reader: Callable[[], int | None] | None = read_throttled
    subset_strategies: dict[str, str] = field(default_factory=dict)
    sleep: Callable[[float], None] = time.sleep


@dataclass
class RunSummary:
    blocks: int = 0
    inserted: int = 0
    skipped_blocks: int = 0
    errors: int = 0
    throttled_reruns: int = 0
    warmups: int = 0
    order: list[str] = field(default_factory=list)


def block_order(configs: Sequence[ExpConfig], seed: int) -> list[ExpConfig]:
    """Seeded shuffle of config blocks, independent of the input order."""
    ordered = sorted(configs, key=lambda c: c.config_id)
    random.Random(seed).shuffle(ordered)
    return ordered


class Runner:
    def __init__(self, conn: sqlite3.Connection, backend: Backend, session: TelemetrySession, opts: RunOptions):
        self.conn = conn
        self.backend = backend
        self.session = session
        self.opts = opts
        self._items: dict[str, list[dict]] = {}
        self._rankings: dict[str, dict[str, dict]] = {}
        self._loaded_path: str | None = None
        self.summary = RunSummary()
        # Distinguishes blocks of separate invocations so a resumed block never overwrites an earlier row.
        self.session_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + secrets.token_hex(2)

    def items_for(self, cfg: ExpConfig) -> list[dict]:
        if cfg.dataset not in self._items:
            self._items[cfg.dataset] = load_split(self.opts.data_dir, cfg.dataset, "test")
        strategy = self.opts.subset_strategies.get(cfg.dataset, "random")
        items = edge_subset(self._items[cfg.dataset], cfg.n_items, strategy, cfg.seed)
        return items[: self.opts.limit] if self.opts.limit else items

    def _retrieved(self, cfg: ExpConfig, item: dict) -> tuple[list[str] | None, int | None]:
        if cfg.context_mode != "retrieved":
            return None, None
        if cfg.dataset not in self._rankings:
            self._rankings[cfg.dataset] = load_rankings(self.opts.data_dir, cfg.dataset)
        ranking = self._rankings[cfg.dataset].get(item["query_id"])
        if ranking is None:
            raise KeyError(f"no retrieval ranking for {cfg.dataset}/{item['query_id']}")
        return top_k(ranking, cfg.k)

    def _count_tokens(self, text: str) -> int:
        n = self.backend.count_tokens(text)
        return n if n is not None else heuristic_token_count(text)

    def _cooldown(self) -> None:
        limit = self.opts.cooldown_below_c
        if limit is None:
            return
        temp = self.session.current_temp()
        waited = 0.0
        while temp is not None and temp >= limit and waited < self.opts.cooldown_max_wait_s:
            log.info("cool-down: %.1f C >= %.1f C, waiting", temp, limit)
            self.opts.sleep(self.opts.cooldown_poll_s)
            waited += self.opts.cooldown_poll_s
            temp = self.session.current_temp()
        if temp is not None and temp >= limit:
            log.warning("cool-down gate timed out at %.1f C; starting block anyway", temp)

    def _ensure_loaded(self, cfg: ExpConfig, items: list[dict]) -> None:
        if self._loaded_path == cfg.model_path and not self.backend.is_mock:
            return
        restarted = self._loaded_path != cfg.model_path
        self.backend.load(cfg)
        self._loaded_path = cfg.model_path
        if restarted:
            for item in items[: self.opts.warmup]:
                try:
                    self.query(cfg, item, block_id=None)
                    self.summary.warmups += 1
                except Exception as exc:
                    log.warning("warm-up query failed: %s", exc)

    def prepare(self, cfg: ExpConfig, item: dict) -> tuple[Prompt, int | None]:
        retrieved, hit = self._retrieved(cfg, item)
        prompt = build_prompt(cfg.dataset, item, cfg.context_mode, retrieved, self.opts.context_tokens,
                              cfg.max_new_tokens, self._count_tokens, self.opts.overhead_tokens)
        return prompt, hit

    def query(self, cfg: ExpConfig, item: dict, block_id: str | None,
              prepared: tuple[Prompt, int | None] | None = None) -> dict[str, Any]:
        prompt, hit = prepared if prepared is not None else self.prepare(cfg, item)
        row: dict[str, Any] = {"config_id": cfg.config_id, "query_id": item["query_id"], "block_id": block_id,
                               "started_at": db.utcnow(), "truncated": int(prompt.truncated),
                               "retrieval_hit": hit, "gold_answer": str(item["gold"]), "is_mock": cfg.is_mock}
        mark = self.session.begin()
        try:
            res = self.backend.generate(prompt.messages, cfg.max_new_tokens, cfg.temperature, cfg.seed)
        except Exception as exc:
            self.session.end(mark)
            row.update(error=f"{type(exc).__name__}: {exc}", parse_ok=0, correct=None)
            return row
        st = self.session.end(mark)
        energy, avg_w = st.energy_j, st.avg_power_w
        if st.peak_mem_mb is None and res.simulated_mem_mb is not None:
            st.peak_mem_mb, st.mem_kind = res.simulated_mem_mb, "mock"
        if res.power_trace:
            ts, ps = zip(*res.power_trace)
            energy = integrate_trapezoid(ts, ps, 0.0, res.e2e_s)
            avg_w = energy / res.e2e_s if energy is not None and res.e2e_s > 0 else None
        throttle_val = self.opts.throttle_reader() if self.opts.throttle_reader else None
        throttled = is_throttled_now(throttle_val)
        sc = score(cfg.dataset, res.text, item)
        row.update(
            prompt_tokens=res.prompt_tokens, output_tokens=res.output_tokens, ttft_s=res.ttft_s, tpot_s=res.tpot_s,
            e2e_latency_s=res.e2e_s, tokens_per_s=res.tokens_per_s, server_prompt_ms=res.server_prompt_ms,
            server_predicted_ms=res.server_predicted_ms, peak_mem_mb=st.peak_mem_mb, mem_kind=st.mem_kind,
            peak_vram_mb=st.peak_vram_mb, energy_j=energy, avg_power_w=avg_w,
            temp_c=st.temp_c if st.temp_c is not None else self.session.current_temp(),
            throttled=None if throttled is None else int(throttled), raw_output=res.text,
            parsed_answer=sc.parsed, parse_ok=int(sc.parse_ok), correct=int(sc.correct), error=None,
        )
        if st.cpu_s is not None and block_id is not None:
            db.upsert_cpu_usage(self.conn, {"scope": "query", "ref_id": f"{cfg.config_id}:{item['query_id']}",
                                            "config_id": cfg.config_id, "device": cfg.device,
                                            "started_at": row["started_at"], "wall_s": st.wall_s, "cpu_s": st.cpu_s,
                                            "peak_mem_mb": st.peak_mem_mb, "mem_kind": st.mem_kind, "n_queries": 1,
                                            "source": "container-cgroup", "coverage": "harness window",
                                            "is_mock": cfg.is_mock})
        if self.opts.keep_raw_telemetry and st.samples:
            db.insert_telemetry(self.conn, "query", f"{cfg.config_id}:{item['query_id']}",
                                [s.as_dict() for s in st.samples])
        return row

    @property
    def metered(self) -> bool:
        return bool(self.opts.block_energy and self.opts.block_energy.startswith("usb_meter"))

    def _read_meter(self, label: str, at_least: float | None = None) -> tuple[float | None, str]:
        """One meter reading and the moment it was taken. CSV mode records only the time; values come at import.

        A cumulative counter never decreases, so a manual value below the previous reading is a typo: re-prompt.
        """
        value = None
        if self.opts.block_energy == "usb_meter_manual" and self.opts.meter_prompt:
            msg = f"{label} enter meter cumulative mWh: "
            while True:
                try:
                    value = float(self.opts.meter_prompt(msg))
                except ValueError:
                    msg = f"{label} not a number; enter meter cumulative mWh: "
                    continue
                if at_least is None or value >= at_least:
                    break
                log.warning("meter reading %.3f below previous %.3f; re-enter", value, at_least)
                msg = f"{label} must be >= {at_least:g} mWh; re-enter: "
        return value, datetime.fromtimestamp(self.opts.clock(), timezone.utc).isoformat(timespec="milliseconds")

    def _block_start_mwh(self, block_id: str) -> float | None:
        r = self.conn.execute("SELECT meter_start_mwh FROM energy_blocks WHERE block_id=?", (block_id,)).fetchone()
        return r[0] if r else None

    def _existing_block(self, cfg: ExpConfig, repeat: int) -> bool:
        return self.conn.execute("SELECT 1 FROM energy_blocks WHERE config_id=? AND repeat_index=? AND ended_at IS NOT NULL",
                                 (cfg.config_id, repeat)).fetchone() is not None

    def block_id(self, cfg: ExpConfig, repeat: int, attempt: int) -> str:
        return f"{cfg.config_id}-r{repeat}-a{attempt}-{self.session_tag}"

    def run_block(self, cfg: ExpConfig, repeat: int, attempt: int) -> bool:
        """Run one block; returns True when it finished without a throttled query."""
        items = self.items_for(cfg)
        if repeat == 0:
            done = db.done_query_ids(self.conn, cfg.config_id, include_errors=not self.opts.retry_errors)
            todo = [i for i in items if i["query_id"] not in done]
        else:
            todo = items
        if not todo:
            self.summary.skipped_blocks += 1
            return True
        self._cooldown()
        self._ensure_loaded(cfg, items)
        prepared = [(item, self.prepare(cfg, item)) for item in todo]
        block_id = self.block_id(cfg, repeat, attempt)
        source = self.opts.block_energy or ("mock" if self.backend.is_mock else self.session.energy_source)
        base = {"block_id": block_id, "config_id": cfg.config_id, "device": cfg.device, "repeat_index": repeat,
                "source": source, "is_mock": cfg.is_mock}
        idle_w = None
        if self.metered:
            i0, ti0 = self._read_meter(f"[{block_id}] IDLE START")
            self.opts.sleep(self.opts.idle_s)
            m0, t_start = self._read_meter(f"[{block_id}] IDLE END / BLOCK START", i0)
            db.upsert_energy_block(self.conn, {**base, "idle_started_at": ti0, "idle_meter_start_mwh": i0,
                                               "started_at": t_start, "meter_start_mwh": m0})
        else:
            if self.opts.idle_s > 0:
                idle_w, idle_src = self.session.measure_idle(self.opts.idle_s)
                if idle_src:
                    db.set_env(self.conn, {f"{cfg.device}/idle_source": idle_src})
            db.upsert_energy_block(self.conn, {**base, "started_at": db.utcnow(), "idle_power_w": idle_w})
        bmark = self.session.begin()
        t0 = time.perf_counter()
        throttled_any, n, mock_energy, mock_dur = False, 0, 0.0, 0.0
        for item, prep in prepared:
            row = self.query(cfg, item, block_id if repeat == 0 else None, prep)
            n += 1
            if row.get("error"):
                self.summary.errors += 1
            throttled_any |= bool(row.get("throttled"))
            if self.backend.is_mock:
                mock_energy += row.get("energy_j") or 0.0
                mock_dur += row.get("e2e_latency_s") or 0.0
            if repeat == 0 and db.insert_run(self.conn, row, replace=self.opts.retry_errors):
                self.summary.inserted += 1
        bst = self.session.end(bmark)
        duration = mock_dur if self.backend.is_mock else time.perf_counter() - t0
        if bst.cpu_s is not None:
            db.upsert_cpu_usage(self.conn, {"scope": "block", "ref_id": block_id, "config_id": cfg.config_id,
                                            "device": cfg.device, "wall_s": bst.wall_s, "cpu_s": bst.cpu_s,
                                            "peak_mem_mb": bst.peak_mem_mb, "mem_kind": bst.mem_kind, "n_queries": n,
                                            "source": "container-cgroup", "coverage": "harness window",
                                            "is_mock": cfg.is_mock})
        if self.metered:
            m1, t_end = self._read_meter(f"[{block_id}] BLOCK END", self._block_start_mwh(block_id))
            db.upsert_energy_block(self.conn, {"block_id": block_id, "ended_at": t_end, "meter_end_mwh": m1,
                                               "n_queries": n})
        else:
            energy = mock_energy if self.backend.is_mock else bst.energy_j
            db.upsert_energy_block(self.conn, {
                "block_id": block_id, "ended_at": db.utcnow(), "n_queries": n, "energy_j": energy,
                "idle_energy_j": idle_w * duration if idle_w is not None else None})
        self.summary.blocks += 1
        if throttled_any:
            log.warning("block %s had throttled queries", block_id)
        return not throttled_any

    def run(self, configs: Sequence[ExpConfig]) -> RunSummary:
        for cfg in configs:
            db.upsert_config(self.conn, cfg.db_row())
        order = block_order(configs, self.opts.block_order_seed)
        self.summary.order = [c.config_id for c in order]
        self.session.start()
        try:
            # Repeat r of every config runs before repeat r + 1, so repeats do not share one thermal state.
            for repeat in range(max(1, self.opts.repeats)):
                for i, cfg in enumerate(order, 1):
                    if repeat > 0 and self._existing_block(cfg, repeat):
                        continue
                    log.info("[repeat %d, %d/%d] %s %s %s %s k=%d on %s", repeat, i, len(order), cfg.model,
                             cfg.strategy, cfg.precision, cfg.dataset, cfg.k, cfg.device)
                    for attempt in range(self.opts.max_block_attempts):
                        ok = self.run_block(cfg, repeat, attempt)
                        if ok or not self.opts.rerun_throttled or attempt + 1 >= self.opts.max_block_attempts:
                            break
                        bid = self.block_id(cfg, repeat, attempt)
                        removed = db.delete_block_runs(self.conn, bid)
                        self.conn.execute("DELETE FROM energy_blocks WHERE block_id=?", (bid,))
                        self.conn.commit()
                        self.summary.inserted -= removed
                        self.summary.throttled_reruns += 1
                        log.info("re-running throttled block %s (%d rows removed)", bid, removed)
        finally:
            self.session.stop()
            self.backend.unload()
        log.info("run finished: %s", {k: v for k, v in self.summary.__dict__.items() if k != "order"})
        return self.summary
