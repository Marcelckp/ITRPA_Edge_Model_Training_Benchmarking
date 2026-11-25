"""`slmbench` command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from . import config as C
from . import db

log = logging.getLogger("slmbench")

DEFAULT_DB = "results/results.db"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=DEFAULT_DB, help="results SQLite DB")
    p.add_argument("--matrix", default=str(C.DEFAULT_MATRIX))
    p.add_argument("--devices-dir", default=str(C.DEFAULT_DEVICES_DIR))


def _tariff_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tariff-file", default=str(C.DEFAULT_TARIFF))
    p.add_argument("--tariff", type=float, help="override ZAR per kWh")
    p.add_argument("--grid-factor", type=float, help="override kgCO2e per kWh")


def _tariff(a: argparse.Namespace) -> dict[str, Any]:
    return C.load_tariff(a.tariff_file, a.tariff, a.grid_factor)


def cmd_env(a: argparse.Namespace) -> int:
    from .env import collect

    dev = C.load_device(a.device, a.devices_dir) if a.device else {"name": "unknown", "paths": {}}
    items = collect(dev["name"], a.llama_cpp_dir or dev.get("paths", {}).get("llama_cpp_dir"), a.note)
    if a.ambient_c is not None:
        items[f"{dev['name']}/ambient_c/{db.utcnow()}"] = a.ambient_c
    conn = db.connect(a.db)
    db.set_env(conn, items)
    print(json.dumps(items, indent=2, default=str))
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    from .datasets import preparer

    m = C.load_matrix(a.matrix)
    dcfg = m["datasets"][a.dataset]
    tcfg = m.get("training", {}).get("datasets", {}).get(a.dataset, {})
    kwargs = dict(labeled=a.labeled, artificial=a.artificial, train=a.train, dev=a.dev, test=a.test,
                  test_ids_file=a.test_ids or dcfg.get("test_ids_file"), n_test=int(dcfg.get("n_test", 500)),
                  n_train=tcfg.get("n_train"), class_balanced=bool(tcfg.get("class_balanced", False)),
                  seed=int(m.get("seed", 42)), use_hf=a.hf, allow_unofficial_split=a.allow_unofficial_split)
    info = preparer(a.dataset)(a.out, **kwargs)
    print(json.dumps(info, indent=2))
    return 0


def cmd_build_index(a: argparse.Namespace) -> int:
    from .rag import build

    m = C.load_matrix(a.matrix)
    info = build(a.dataset, a.data_dir, max(m.get("rag_k", [10])))
    print(json.dumps(info, indent=2))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from . import train as T

    m = C.load_matrix(a.matrix)
    if a.list_jobs:
        jobs = C.training_plan(m, a.list_jobs)
        print(json.dumps(jobs, indent=2))
        print(f"{len(jobs)} jobs", file=sys.stderr)
        return 0
    conn = db.connect(a.db)
    if a.select:
        chosen = T.select_learning_rates(conn, int(m["training"].get("selection_seed", 42)),
                                         allow_mock=a.allow_mock)
        print(json.dumps(chosen, indent=2))
        return 0
    if not (a.model and a.strategy and a.dataset):
        print("train needs --model --strategy --dataset (or --list-jobs / --select)", file=sys.stderr)
        return 2
    seed = a.seed if a.seed is not None else int(m["training"].get("selection_seed", 42))
    if a.lr == "selected":
        lr = T.selected_lr(conn, a.model, a.strategy, a.dataset)
        if lr is None:
            print("no selected learning rate yet; run `slmbench train --select` after the selection runs",
                  file=sys.stderr)
            return 2
    else:
        lr = float(a.lr)
    plan = T.make_plan(m, a.model, a.strategy, a.dataset, seed, lr, a.data_dir, a.models_dir)
    if a.per_device_batch_size or a.grad_accum:
        if not a.bench:
            print("--per-device-batch-size/--grad-accum change the recipe; they are for --bench only", file=sys.stderr)
            return 2
        plan["per_device_batch_size"] = a.per_device_batch_size or plan["per_device_batch_size"]
        plan["gradient_accumulation_steps"] = a.grad_accum or plan["gradient_accumulation_steps"]
        plan["effective_batch_size"] = plan["per_device_batch_size"] * plan["gradient_accumulation_steps"]
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if a.token_stats:
        print(json.dumps(T.token_stats(plan, a.max_train_examples)))
        return 0
    if a.mock:
        tid = db.insert_training_run(conn, T.mock_training_row(plan))
    elif a.bench:
        limits = T.TrainLimits(max_steps=a.max_steps, time_limit_s=a.time_limit_s,
                               max_train_examples=a.max_train_examples, skip_validation=a.skip_validation,
                               skip_save=True)
        if not limits.capped:
            print("--bench needs --max-steps, --time-limit-s or --max-train-examples", file=sys.stderr)
            return 2
        res = T.run_training(plan, conn, val_limit=a.val_limit, device=a.compute, threads=a.threads, limits=limits)
        print(json.dumps({"bench": True, "planning_only": True, **res}, default=str))
        return 0
    else:
        gpu = C.load_device(a.device, a.devices_dir) if a.device and a.compute != "cpu" else {}
        tid = T.run_training(plan, conn, nvml_index=int(gpu.get("telemetry", {}).get("gpu_index", 0)),
                             val_limit=a.val_limit, device=a.compute, threads=a.threads)
    print(json.dumps({"train_id": tid, "output_dir": plan["output_dir"]}))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from .export import export

    m = C.load_matrix(a.matrix)
    conn = db.connect(a.db) if (not a.dry_run or Path(a.db).exists()) else None
    precisions = tuple(a.precision) if a.precision else ("F16", "Q8_0", "Q4_K_M")
    plan = export(m, conn, a.model, a.strategy, a.dataset, a.trained_dir, a.llama_cpp_dir, a.models_dir,
                  precisions=precisions, dry_run=a.dry_run, skip_existing=a.skip_existing)
    print(json.dumps(plan, indent=2, default=str))
    return 0


def _configs(a: argparse.Namespace, m: dict, is_mock: bool = False) -> list[C.ExpConfig]:
    plans = a.plan or None
    devices = [a.device] if getattr(a, "device", None) else None
    names = {d for p in (plans or m["plans"]) for d in m["plans"][p]["devices"]}
    devs = C.load_devices(names, a.devices_dir)
    return C.expand(m, plans, devs, devices, is_mock=is_mock, filters=C.parse_filters(getattr(a, "filter", None)))


def cmd_estimate(a: argparse.Namespace) -> int:
    m = C.load_matrix(a.matrix)
    cfgs = _configs(a, m, is_mock=a.mock)
    counts = C.count_runs(cfgs)
    out: dict[str, Any] = {"counts": counts, "adapted_models": C.adapted_model_count(m),
                           "training_selection_runs": len(C.training_plan(m, "selection")),
                           "training_variability_runs": len(C.training_plan(m, "variability"))}
    if not a.counts_only:
        from .estimate import estimate

        r = m.get("runner", {})
        conn = db.connect(a.db)
        out["projection"] = estimate(conn, cfgs, int(r.get("warmup_queries", 3)), float(r.get("idle_baseline_s", 60)),
                                     a.startup_s, a.cooldown_s, int(r.get("energy_repeats", 1)))
    print(json.dumps(out, indent=2))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .backends import make_backend
    from .runner import Runner, RunOptions
    from .telemetry import build_session

    m = C.load_matrix(a.matrix)
    if not a.device:
        print("run needs --device", file=sys.stderr)
        return 2
    dev = C.load_device(a.device, a.devices_dir)
    mock = a.backend == "mock"
    cfgs = _configs(a, m, is_mock=mock)
    if not cfgs:
        print("no configurations match", file=sys.stderr)
        return 2
    engines = {c.engine for c in cfgs}
    if len(engines) > 1:
        print(f"one engine per run invocation, got {sorted(engines)}; pass --plan", file=sys.stderr)
        return 2
    engine = engines.pop()
    settings = dict(dev["backends"].get(engine, {}))
    settings["_device_paths"] = {k: str(Path(v).expanduser()) for k, v in dev.get("paths", {}).items()}
    if a.server_log:
        settings["server_log"] = a.server_log
    settings["chat_template_kwargs"] = dict(m.get("chat_template_kwargs") or {})
    backend = make_backend("mock" if mock else settings.get("kind", engine), settings)
    tel = dev.get("telemetry", {})
    session = build_session(tel, lambda: backend.server_pid, mock=mock)
    r = m.get("runner", {})
    runner_dev = dev.get("runner", {})
    opts = RunOptions(
        data_dir=a.data_dir, warmup=a.warmup if a.warmup is not None else int(r.get("warmup_queries", 3)),
        idle_s=a.idle_s if a.idle_s is not None else float(r.get("idle_baseline_s", 60)),
        repeats=a.repeats if a.repeats is not None else int(r.get("energy_repeats", 3)),
        block_order_seed=int(r.get("block_order_seed", 42)),
        cooldown_below_c=None if a.no_cooldown else float(runner_dev.get("cooldown_below_c", r.get("cooldown_below_c", 50))),
        cooldown_max_wait_s=float(runner_dev.get("thermal_wait_max_s", 900)),
        rerun_throttled=bool(r.get("rerun_throttled", True)), limit=a.pilot or a.limit, retry_errors=a.retry_errors,
        context_tokens=int(m.get("context_tokens", 4096)), keep_raw_telemetry=a.keep_telemetry,
        overhead_tokens=int(m.get("prompt_overhead_tokens", 64)),
        block_energy=None if mock else (a.block_energy or tel.get("block_energy")),
        meter_prompt=lambda msg: float(input(msg)),
        subset_strategies={k: v.get("subset_strategy", "random") for k, v in m["datasets"].items()},
    )
    if mock:
        opts.throttle_reader = None
    conn = db.connect(a.db)
    summary = Runner(conn, backend, session, opts).run(cfgs)
    print(json.dumps({k: v for k, v in summary.__dict__.items() if k != "order"}, indent=2))
    return 0


def cmd_energy_block(a: argparse.Namespace) -> int:
    conn = db.connect(a.db)
    if a.action == "start":
        bid = a.block_id or f"{a.config_id}-r{a.repeat}-manual"
        mwh = a.meter_mwh if a.meter_mwh is not None else float(input("meter mWh at START: "))
        db.upsert_energy_block(conn, {"block_id": bid, "config_id": a.config_id, "device": a.device,
                                      "repeat_index": a.repeat, "started_at": db.utcnow(), "meter_start_mwh": mwh,
                                      "idle_power_w": a.idle_power_w, "source": "usb_meter_manual"})
        print(bid)
    elif a.action == "stop":
        row = conn.execute("SELECT * FROM energy_blocks WHERE block_id=?", (a.block_id,)).fetchone()
        if row is None:
            print(f"unknown block {a.block_id}", file=sys.stderr)
            return 2
        n = a.n_queries
        if n is None:
            n = conn.execute("SELECT COUNT(*) FROM runs WHERE block_id=?", (a.block_id,)).fetchone()[0] or None
        mwh = a.meter_mwh if a.meter_mwh is not None else float(input("meter mWh at END: "))
        db.upsert_energy_block(conn, {"block_id": a.block_id, "ended_at": db.utcnow(), "n_queries": n,
                                      "meter_end_mwh": mwh})
    elif a.action == "import-csv":
        from .telemetry.usb_meter_csv import block_readings, read_meter_csv

        try:
            rows = read_meter_csv(a.csv, a.tz)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if not a.device:
            print("import-csv needs --device", file=sys.stderr)
            return 2
        # Only meter-mode blocks: NVML/RAPL/tegrastats/PMIC blocks keep their own instrument.
        q = "SELECT * FROM energy_blocks WHERE ended_at IS NOT NULL AND device=? AND source LIKE 'usb_meter%'"
        n = 0
        for b in conn.execute(q, (a.device,)).fetchall():
            rd = block_readings(rows, b["started_at"], b["ended_at"], b["idle_started_at"])
            if rd["meter_start_mwh"] is None or rd["meter_end_mwh"] is None:
                continue
            db.upsert_energy_block(conn, {"block_id": b["block_id"], "source": "usb_meter_csv",
                                          "idle_power_w": None, "idle_energy_j": None, **rd})
            n += 1
        print(f"updated {n} blocks from {a.csv}")
    elif a.action == "list":
        for b in conn.execute("SELECT * FROM energy_blocks ORDER BY started_at").fetchall():
            print(json.dumps(dict(b)))
    return 0


def cmd_energy_estimate(a: argparse.Namespace) -> int:
    from . import energy_estimate as E

    conn = db.connect(a.db)
    model = E.load_model(a.model_file)
    out: dict[str, Any] = {"energy_basis": model.energy_basis, "label": model.label}
    for csv_path in a.cgroup_csv or []:
        if not a.device:
            print("--cgroup-csv needs --device (the device whose query/block windows the samples cover)", file=sys.stderr)
            return 2
        out[f"joined:{csv_path}"] = E.import_host_csv(conn, csv_path, a.device)
    out["estimates"] = E.compute(conn, model, mem_gb_override=a.mem_gb)
    if a.out:
        tables = E.summary_tables(conn, _tariff(a), allow_mock=a.allow_mock)
        caption = (C.load_matrix(a.matrix).get("report") or {}).get("caption_note") if a.matrix else None
        out["files"] = E.write_outputs(tables, model, _tariff(a), a.out, caption)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .analysis import MockDataRefused, analyze

    m = C.load_matrix(a.matrix)
    try:
        res = analyze(db.connect(a.db), m, _tariff(a), a.out, allow_mock=a.allow_mock)
    except MockDataRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({k: len(v) for k, v in res.items()}, indent=2))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .report import ReportRefused, build_report

    m = C.load_matrix(a.matrix)
    try:
        man = build_report(db.connect(a.db), m, _tariff(a), a.out, allow_mock=a.allow_mock)
    except ReportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"is_mock": man["is_mock"], "files": len(man["files"])}, indent=2))
    return 0


def cmd_advise(a: argparse.Namespace) -> int:
    from .advise import Constraints, advise
    from .analysis import MockDataRefused

    m = C.load_matrix(a.matrix)
    t = _tariff(a)
    c = Constraints(device=a.device, ram_mb=a.ram_mb, latency_budget_s=a.latency_budget_s,
                    monthly_queries=a.monthly_queries, min_accuracy=a.min_accuracy, dataset=a.dataset,
                    tariff_zar_per_kwh=t["tariff_zar_per_kwh"], grid_kgco2e_per_kwh=t["grid_kgco2e_per_kwh"],
                    amortise_months=a.amortise_months, objective=a.objective)
    try:
        res = advise(db.connect(a.db), m, c, a.out, t["tariff_source"], allow_mock=a.allow_mock)
    except MockDataRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    top = res["feasible"].head(5)
    print(top[["rank", "model", "variant", "precision", "dataset", "accuracy", "p95_e2e_s", "monthly_zar"]].to_string(index=False)
          if not top.empty else "no feasible configuration")
    print(f"HTML: {res['html']}")
    return 0


def cmd_check_template(a: argparse.Namespace) -> int:
    from .backends import make_backend
    from .datasets import load_split
    from .prompts import build_prompt, render
    from .template_check import compare

    m = C.load_matrix(a.matrix)
    dev = C.load_device(a.device, a.devices_dir)
    cfgs = [c for c in _configs(a, m) if c.engine == "llamacpp" and c.context_mode != "retrieved"]
    if not cfgs:
        print("no llamacpp configuration matches", file=sys.stderr)
        return 2
    cfg = cfgs[0]
    settings = dict(dev["backends"]["llamacpp"])
    settings["_device_paths"] = {k: str(Path(v).expanduser()) for k, v in dev.get("paths", {}).items()}
    kwargs = dict(m.get("chat_template_kwargs") or {})
    settings["chat_template_kwargs"] = kwargs
    backend = make_backend("llamacpp", settings)
    item = load_split(a.data_dir, cfg.dataset, "test")[0]
    prompt = build_prompt(cfg.dataset, item, cfg.context_mode, None, int(m.get("context_tokens", 4096)),
                          cfg.max_new_tokens, overhead_tokens=int(m.get("prompt_overhead_tokens", 64)))
    from transformers import AutoTokenizer  # lazy: [gpu] extra or a standalone transformers install

    trained = m["strategies"][cfg.strategy].get("trained")
    tok_src = a.tokenizer or (C.merged_model_dir(m, cfg.model, cfg.strategy, cfg.train_dataset) if trained else cfg.hf_id)
    tok = AutoTokenizer.from_pretrained(tok_src)
    backend.load(cfg)
    try:
        res = compare(render(prompt.messages, tok, template_kwargs=kwargs), backend.apply_template(prompt.messages),
                      tok.bos_token)
    finally:
        backend.unload()
    print(json.dumps({"config_id": cfg.config_id, "model_path": cfg.model_path, "match": res["match"]}, indent=2))
    if not res["match"]:
        print(res["diff"], file=sys.stderr)
        return 1
    return 0


def cmd_preflight(a: argparse.Namespace) -> int:
    from .preflight import FAIL, format_table, run_preflight

    m = C.load_matrix(a.matrix)
    dev = C.load_device(a.device, a.devices_dir)
    plans = a.plan or [n for n, p in m["plans"].items() if a.device in p["devices"]]
    cfgs = C.expand(m, plans, {a.device: dev}, [a.device], filters=C.parse_filters(a.filter)) if plans else []
    models_dir = dev.get("paths", {}).get("models_dir") or m.get("paths", {}).get("models_dir", "models")
    checks = run_preflight(dev, cfgs, a.db, models_dir, meter_csv=a.meter_csv, tz=a.tz, verify_sha=not a.no_sha,
                           max_temp_c=float(dev.get("runner", {}).get("cooldown_below_c", 50)),
                           warn_gb=a.warn_free_gb, fail_gb=a.min_free_gb)
    print(format_table(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


def cmd_merge(a: argparse.Namespace) -> int:
    counts = db.merge_dbs(a.into, a.sources)
    print(json.dumps(counts, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slmbench", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env", help="record software/hardware environment")
    _common(p)
    p.add_argument("--device")
    p.add_argument("--llama-cpp-dir")
    p.add_argument("--note")
    p.add_argument("--ambient-c", type=float, help="ambient temperature for this session")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("prepare-data", help="normalise raw dataset files into data/<dataset>/*.jsonl")
    _common(p)
    p.add_argument("--dataset", required=True, choices=["pubmedqa", "finqa", "arc_easy"])
    p.add_argument("--out", default="data")
    for f in ("labeled", "artificial", "train", "dev", "test", "test-ids"):
        p.add_argument(f"--{f}")
    p.add_argument("--hf", action="store_true", help="download from Hugging Face (needs `datasets`)")
    p.add_argument("--allow-unofficial-split", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("build-index", help="BM25 index + precomputed top-K rankings")
    _common(p)
    p.add_argument("--dataset", required=True, choices=["pubmedqa", "finqa"])
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("train", help="FFT/LoRA/QLoRA training run, LR selection, or job listing")
    _common(p)
    p.add_argument("--model")
    p.add_argument("--strategy", choices=["fft", "lora", "qlora", "celora"])
    p.add_argument("--dataset", choices=["pubmedqa", "finqa"])
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", default="selected", help="learning rate from the grid, or 'selected'")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models-dir")
    p.add_argument("--val-limit", type=int)
    p.add_argument("--dry-run", action="store_true", help="validate and print the plan without importing torch")
    p.add_argument("--mock", action="store_true", help="write a simulated is_mock=1 row (tests only)")
    p.add_argument("--select", action="store_true", help="mark the best-validation LR per adapted model")
    p.add_argument("--allow-mock", action="store_true", help="let --select consider is_mock rows (tests only)")
    p.add_argument("--device", default="gpu-workstation", help="device profile (NVML index)")
    p.add_argument("--list-jobs", choices=["selection", "variability"])
    p.add_argument("--compute", choices=["auto", "cuda", "cpu"], default="auto",
                   help="training device; auto = CUDA when available, else CPU (bf16 autocast if the CPU has AVX512_BF16/AMX)")
    p.add_argument("--threads", type=int, help="torch CPU threads (CPU training)")
    p.add_argument("--bench", action="store_true",
                   help="capped feasibility run: prints throughput JSON, writes nothing to the DB (planning only)")
    p.add_argument("--max-steps", type=int, help="--bench: optimiser steps")
    p.add_argument("--time-limit-s", type=float, help="--bench: stop after this many training seconds (>= 3 steps)")
    p.add_argument("--max-train-examples", type=int, help="--bench / --token-stats: first N training examples")
    p.add_argument("--per-device-batch-size", type=int, help="--bench only")
    p.add_argument("--grad-accum", type=int, help="--bench only")
    p.add_argument("--skip-validation", action="store_true", help="--bench: no validation generation")
    p.add_argument("--token-stats", action="store_true", help="token counts of the training set as tokenised for training")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export-gguf", help="merge adapters and export GGUF F16/Q8_0/Q4_K_M")
    _common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--trained-dir")
    p.add_argument("--llama-cpp-dir", default="~/llama.cpp")
    p.add_argument("--models-dir")
    p.add_argument("--precision", action="append", choices=["F16", "Q8_0", "Q4_K_M"],
                   help="precision to export and record (repeatable; default all three; F16 is always built)")
    p.add_argument("--skip-existing", action="store_true",
                   help="keep GGUF files already on disk (still hashed and recorded in --db)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("estimate", help="planned counts and projected wall-clock from pilot runs")
    _common(p)
    p.add_argument("--plan", action="append")
    p.add_argument("--device")
    p.add_argument("--filter", action="append")
    p.add_argument("--counts-only", action="store_true")
    p.add_argument("--mock", action="store_true", help="estimate from mock pilot config ids")
    p.add_argument("--startup-s", type=float, default=20.0)
    p.add_argument("--cooldown-s", type=float, default=0.0)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("run", help="execute config blocks on this device")
    _common(p)
    p.add_argument("--plan", action="append")
    p.add_argument("--device")
    p.add_argument("--backend", choices=["auto", "mock"], default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--filter", action="append", help="key=value[,value] e.g. model=qwen2.5-0.5b")
    p.add_argument("--limit", type=int)
    p.add_argument("--pilot", type=int, help="first N items per config (for `estimate`)")
    p.add_argument("--warmup", type=int)
    p.add_argument("--idle-s", type=float)
    p.add_argument("--repeats", type=int, help="energy repeat blocks per config")
    p.add_argument("--block-energy", choices=["usb_meter_manual", "usb_meter_csv"])
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--no-cooldown", action="store_true")
    p.add_argument("--keep-telemetry", action="store_true")
    p.add_argument("--server-log")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("energy-block", help="metered energy blocks: start/stop/import-csv/list")
    _common(p)
    p.add_argument("action", choices=["start", "stop", "import-csv", "list"])
    p.add_argument("--block-id")
    p.add_argument("--config-id")
    p.add_argument("--device")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--meter-mwh", type=float)
    p.add_argument("--idle-power-w", type=float)
    p.add_argument("--n-queries", type=int)
    p.add_argument("--csv")
    p.add_argument("--tz", help="meter clock zone for timestamps without an offset, e.g. +02:00 or Africa/Johannesburg")
    p.set_defaults(func=cmd_energy_block)

    p = sub.add_parser("energy-estimate",
                       help="ESTIMATED energy from container CPU time (Green Algorithms model), not metered")
    _common(p)
    _tariff_args(p)
    p.add_argument("--cgroup-csv", action="append",
                   help="host cgroup samples (scripts/pilot/cgroup_logger.py) to join to --device's query/block windows")
    p.add_argument("--device", help="device whose runs the --cgroup-csv samples cover")
    p.add_argument("--model-file", default=str(C.DEFAULT_CONFIG_DIR / "energy-model.yaml"))
    p.add_argument("--mem-gb", type=float, help="memory term in GB (default: peak cgroup usage of each window)")
    p.add_argument("--out", help="write estimated_energy.md and CSVs here")
    p.add_argument("--allow-mock", action="store_true")
    p.set_defaults(func=cmd_energy_estimate)

    p = sub.add_parser("analyze", help="tidy CSVs and hypothesis tests")
    _common(p)
    _tariff_args(p)
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--allow-mock", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="Markdown tables, PNG figures, hypothesis_decisions.md")
    _common(p)
    _tariff_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-mock", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("advise", help="rank feasible configs; static HTML")
    _common(p)
    _tariff_args(p)
    p.add_argument("--device", required=True)
    p.add_argument("--ram-mb", type=float)
    p.add_argument("--latency-budget-s", type=float)
    p.add_argument("--monthly-queries", type=int, default=10000)
    p.add_argument("--min-accuracy", type=float)
    p.add_argument("--dataset")
    p.add_argument("--amortise-months", type=int, default=12)
    p.add_argument("--objective", choices=["cost", "accuracy", "latency"], default="cost")
    p.add_argument("--out", default="results/advisor.html")
    p.add_argument("--allow-mock", action="store_true")
    p.set_defaults(func=cmd_advise)

    p = sub.add_parser("check-template", help="compare the training chat template with llama-server /apply-template")
    _common(p)
    p.add_argument("--device", required=True)
    p.add_argument("--plan", action="append")
    p.add_argument("--filter", action="append")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tokenizer", help="HF id or local dir (default: the config's HF model or merged dir)")
    p.set_defaults(func=cmd_check_template)

    p = sub.add_parser("preflight", help="read-only readiness check before a campaign (exit 1 on any FAIL)")
    _common(p)
    p.add_argument("--device", required=True)
    p.add_argument("--plan", action="append", help="plans whose model files to check (default: all on this device)")
    p.add_argument("--filter", action="append", help="key=value[,value]: check only matching configs' model files")
    p.add_argument("--meter-csv", help="USB meter CSV being logged (default: telemetry.meter_csv)")
    p.add_argument("--tz", help="meter clock zone for naive timestamps (default: telemetry.meter_tz)")
    p.add_argument("--no-sha", action="store_true", help="skip sha256 verification of model files")
    p.add_argument("--warn-free-gb", type=float, default=20.0)
    p.add_argument("--min-free-gb", type=float, default=2.0)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("merge-db", help="merge results DBs from several devices")
    p.add_argument("--into", required=True)
    p.add_argument("sources", nargs="+")
    p.set_defaults(func=cmd_merge)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
