"""Merge adapters into the bf16 base and export GGUF F16 / Q8_0 / Q4_K_M with llama.cpp tools."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import db
from .config import ConfigError, merged_model_dir, resolve_model_path, trained_model_dir

log = logging.getLogger(__name__)

QUANTS = ("Q8_0", "Q4_K_M")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gguf_commands(hf_dir: str, out_paths: dict[str, str], llama_cpp_dir: str,
                  python: str = sys.executable) -> list[list[str]]:
    """Commands to produce every requested precision; F16 is always converted first."""
    lcpp = os.path.expanduser(llama_cpp_dir)
    quantize = str(Path(lcpp) / "build" / "bin" / "llama-quantize")
    f16 = out_paths["F16"]
    # convert_hf_to_gguf.py defaults to bf16, so f16 must be explicit.
    cmds = [[python, str(Path(lcpp) / "convert_hf_to_gguf.py"), hf_dir, "--outtype", "f16", "--outfile", f16]]
    for q in QUANTS:
        if q in out_paths:
            cmds.append([quantize, f16, out_paths[q], q])
    return cmds


def merge_adapter(base_hf_id: str, adapter_dir: str, out_dir: str, strategy: str) -> str:
    """Write a bf16 merged model; FFT checkpoints are re-saved unchanged in bf16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_dir if strategy == "fft" else base_hf_id)
    if strategy == "fft":
        model = AutoModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.bfloat16)
    else:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(base_hf_id, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    log.info("merged %s -> %s", adapter_dir, out_dir)
    return out_dir


def selected_run(conn: sqlite3.Connection, model: str, strategy: str, dataset: str,
                 seed: int = 42) -> sqlite3.Row | None:
    return conn.execute("SELECT train_id, learning_rate FROM training_runs WHERE model=? AND strategy=? AND dataset=? "
                        "AND seed=? AND selected=1 AND is_mock=0", (model, strategy, dataset, seed)).fetchone()


def verify_provenance(matrix: dict, conn: sqlite3.Connection, model: str, strategy: str, dataset: str,
                      trained_dir: str, models_dir: str | None = None) -> int:
    """Only the seed-42 model at the selected LR may be exported (DESIGN.md 3.1); returns its train_id."""
    seed = int(matrix.get("training", {}).get("selection_seed", 42))
    run = selected_run(conn, model, strategy, dataset, seed)
    if run is None:
        raise ConfigError(f"no selected seed-{seed} training run for {model}/{strategy}/{dataset}; run `train --select`")
    expected = trained_model_dir(matrix, model, strategy, dataset, float(run["learning_rate"]), seed, models_dir)
    if Path(os.path.expanduser(trained_dir)).resolve() != Path(expected).resolve():
        raise ConfigError(f"--trained-dir {trained_dir} is not the selected run ({expected})")
    return int(run["train_id"])


def record_artifact(conn: sqlite3.Connection, model: str, strategy: str, dataset: str, precision: str,
                    path: str | Path, train_id: int | None = None) -> dict[str, Any]:
    p = Path(path)
    row = {"model": model, "strategy": strategy, "dataset": dataset, "precision": precision, "path": str(p),
           "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "train_id": train_id}
    known = conn.execute("SELECT 1 FROM artifacts WHERE path=? AND sha256=?", (row["path"], row["sha256"])).fetchone()
    if known is None:
        db.insert_artifact(conn, row)
    return row


def _output_index(cmd: list[str]) -> int:
    return cmd.index("--outfile") + 1 if "--outfile" in cmd else 2


def _cmd_precision(cmd: list[str]) -> str:
    return "F16" if "--outfile" in cmd else cmd[3]


def _run_atomic(cmd: list[str]) -> None:
    """Write to <out>.part and rename, so an interrupted run never leaves a file that looks complete."""
    i = _output_index(cmd)
    final = cmd[i]
    part = final + ".part"
    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd[:i] + [part] + cmd[i + 1:], check=True)
    os.replace(part, final)


def export(matrix: dict, conn: sqlite3.Connection | None, model_id: str, strategy: str, dataset: str,
           trained_dir: str | None, llama_cpp_dir: str, models_dir: str | None = None,
           precisions: tuple[str, ...] = ("F16", "Q8_0", "Q4_K_M"), dry_run: bool = False,
           skip_existing: bool = False) -> dict[str, Any]:
    models = {m["id"]: m for m in matrix["models"]}
    model = models[model_id]
    trained = matrix["strategies"][strategy].get("trained", False)
    dev = {"paths": {"models_dir": models_dir}} if models_dir else None
    ds = dataset if trained else ""
    wanted = tuple(dict.fromkeys(precisions))
    # F16 is always produced: it is the input of llama-quantize.
    outs = {p: resolve_model_path(matrix, dev, model, strategy if trained else "base", p, ds)
            for p in dict.fromkeys(("F16", *wanted))}
    pending = {p for p, path in outs.items() if not (skip_existing and Path(path).exists())}
    if trained:
        if not trained_dir:
            raise ValueError("--trained-dir is required for trained strategies (the selected-LR seed-42 run)")
        hf_dir = merged_model_dir(matrix, model_id, strategy, dataset, models_dir)
    else:
        hf_dir = model["hf_id"]
    train_id = None
    if trained and conn is not None:
        train_id = verify_provenance(matrix, conn, model_id, strategy, dataset, trained_dir, models_dir)
    cmds = gguf_commands(hf_dir, outs, llama_cpp_dir)
    plan = {"model": model_id, "strategy": strategy, "dataset": ds, "merged_dir": hf_dir,
            "outputs": {p: outs[p] for p in wanted}, "commands": cmds, "train_id": train_id,
            "skipped": sorted(set(outs) - pending)}
    if dry_run:
        return plan
    if "F16" in pending:
        if trained:
            merge_adapter(model["hf_id"], trained_dir, hf_dir, strategy)
        elif not Path(hf_dir).exists():
            from huggingface_hub import snapshot_download  # installed with transformers

            hf_dir = snapshot_download(model["hf_id"])
    cmds = [c for c in gguf_commands(hf_dir, outs, llama_cpp_dir) if _cmd_precision(c) in pending]
    plan["commands"] = cmds
    for path in outs.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    for cmd in cmds:
        _run_atomic(cmd)
    plan["artifacts"] = []
    if conn is not None:
        for prec in wanted:
            plan["artifacts"].append(record_artifact(conn, model_id, strategy if trained else "base", ds, prec,
                                                    outs[prec], train_id))
    return plan
