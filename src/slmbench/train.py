"""FFT / LoRA / QLoRA adaptation (DESIGN.md 3.3). torch, transformers, peft and bitsandbytes import lazily."""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from .config import ConfigError, trained_model_dir
from .datasets import read_jsonl
from .prompts import build_prompt, render, target_text

log = logging.getLogger(__name__)


def make_plan(matrix: dict, model_id: str, strategy: str, dataset: str, seed: int, learning_rate: float,
              data_dir: str | Path, models_dir: str | None = None) -> dict[str, Any]:
    tr = matrix.get("training", {})
    models = {m["id"]: m for m in matrix["models"]}
    if model_id not in models:
        raise ConfigError(f"unknown model {model_id}")
    sdef = matrix["strategies"].get(strategy)
    if not sdef or not sdef.get("trained"):
        raise ConfigError(f"strategy {strategy} is not a training strategy")
    if dataset not in tr.get("datasets", {}):
        raise ConfigError(f"no training config for dataset {dataset}")
    grid = [float(x) for x in tr.get("lr_grid", {}).get(strategy, [])]
    if learning_rate not in grid:
        raise ConfigError(f"learning rate {learning_rate:g} not in the pre-declared grid {grid} for {strategy}")
    eff, per = int(tr.get("effective_batch_size", 16)), int(tr.get("per_device_batch_size", 4))
    if eff % per:
        raise ConfigError("effective_batch_size must be a multiple of per_device_batch_size")
    plan = {
        "model": model_id, "hf_id": models[model_id]["hf_id"], "strategy": strategy, "dataset": dataset,
        "seed": seed, "learning_rate": learning_rate, "epochs": float(tr.get("epochs", 1)),
        "per_device_batch_size": per, "gradient_accumulation_steps": eff // per, "effective_batch_size": eff,
        "lr_scheduler": tr.get("lr_scheduler", "cosine"), "warmup_ratio": float(tr.get("warmup_ratio", 0.03)),
        "weight_decay": float(tr.get("weight_decay", 0.0)), "max_seq_len": int(tr.get("max_seq_len", 4096)),
        "optimizer": tr.get("optimizer", "adamw_torch"),
        "fft_master_dtype": tr.get("fft_master_dtype", "float32"),
        "qlora_upcast_non_quantized": bool(tr.get("qlora_upcast_non_quantized", True)),
        "logging_steps": int(tr.get("logging_steps", 10)),
        "chat_template_kwargs": dict(matrix.get("chat_template_kwargs") or {}),
        "gradient_checkpointing": bool(tr.get("gradient_checkpointing", True)),
        "lora": tr.get("lora", {}) if strategy in ("lora", "qlora", "celora") else None,
        "quantization": tr.get("qlora", {}) if strategy == "qlora" else None,
        "train_file": str(Path(data_dir) / dataset / "train.jsonl"),
        "max_train_examples": (tr["datasets"][dataset] or {}).get("max_train_examples"),
        "val_file": str(Path(data_dir) / dataset / "val.jsonl"),
        "val_metric": tr.get("val_metric", "accuracy"),
        "max_new_tokens": int(matrix["max_new_tokens"][dataset]),
        "context_tokens": int(matrix.get("context_tokens", 4096)),
        "prompt_overhead_tokens": int(matrix.get("prompt_overhead_tokens", 64)),
        "output_dir": trained_model_dir(matrix, model_id, strategy, dataset, learning_rate, seed, models_dir),
    }
    if strategy == "celora":
        raise ConfigError("celora needs a verified public implementation; not wired (report as a limitation)")
    return plan


def native_bf16(torch_mod: Any) -> bool:
    """True only for hardware bf16 (compute capability >= 8); emulated bf16 on older GPUs does not count."""
    cuda = torch_mod.cuda
    if not cuda.is_available():
        return False
    try:
        return bool(cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return cuda.get_device_capability()[0] >= 8


def cpu_flags(cpuinfo: str | None = None) -> set[str]:
    if cpuinfo is None:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            return set()
    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def cpu_bf16(cpuinfo: str | None = None) -> bool:
    """Hardware bf16 on x86 CPUs: AVX512_BF16 or AMX-BF16. Without it torch emulates bf16 slowly, so fp32 is used."""
    return bool(cpu_flags(cpuinfo) & {"avx512_bf16", "amx_bf16"})


def resolve_device(requested: str, torch_mod: Any) -> str:
    if requested == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if requested not in ("cuda", "cpu"):
        raise ConfigError(f"unknown training device {requested}")
    return requested


def precision_recipe(plan: dict[str, Any], bf16_supported: bool, device: str = "cuda") -> dict[str, Any]:
    """Load dtype, autocast flags and the recipe label stored in training_runs.optimizer.

    FFT keeps fp32 master weights: pure-bf16 weights round away most updates at FFT learning rates.
    Pre-Ampere GPUs have no bf16, so autocast falls back to fp16 there. CPUs have no fp16 autocast path in
    the Trainer, so a CPU without bf16 trains in plain fp32.
    """
    if device == "cpu":
        autocast = "bf16" if bf16_supported else "none"
        half = "bfloat16" if bf16_supported else "float32"
    else:
        autocast = "bf16" if bf16_supported else "fp16"
        half = "bfloat16" if bf16_supported else "float16"
    prefix = "cpu " if device == "cpu" else ""
    if plan["strategy"] == "fft":
        load = plan.get("fft_master_dtype", "float32")
        label = f"{prefix}{plan['optimizer']} {'fp32' if load == 'float32' else load}-master {autocast}-autocast"
    else:
        load = half
        label = f"{prefix}{plan['optimizer']} fp32-adapters {autocast}-autocast"
        if plan["strategy"] == "qlora":
            label += (" qlora-kbit-prep(fp32 upcast of non-quantised params)" if plan.get("qlora_upcast_non_quantized", True)
                      else " qlora-no-upcast")
    return {"load_dtype": load, "bf16": bf16_supported, "fp16": device != "cpu" and not bf16_supported,
            "label": label, "bnb_compute_dtype": half}


def select_learning_rates(conn: sqlite3.Connection, selection_seed: int = 42,
                          allow_mock: bool = False) -> list[dict[str, Any]]:
    """Mark, per (model, strategy, dataset), the selection-seed run with the best validation metric.

    Ties go to the smaller learning rate so the choice is reproducible.
    """
    rows = conn.execute(
        "SELECT train_id, model, strategy, dataset, learning_rate, val_metric FROM training_runs "
        "WHERE seed=? AND val_metric IS NOT NULL" + ("" if allow_mock else " AND is_mock=0"),
        (selection_seed,)).fetchall()
    best: dict[tuple, Any] = {}
    for r in rows:
        key = (r["model"], r["strategy"], r["dataset"])
        cur = best.get(key)
        if cur is None or (r["val_metric"], -r["learning_rate"]) > (cur["val_metric"], -cur["learning_rate"]):
            best[key] = r
    conn.execute("UPDATE training_runs SET selected=0 WHERE seed=?", (selection_seed,))
    for r in best.values():
        conn.execute("UPDATE training_runs SET selected=1 WHERE train_id=?", (r["train_id"],))
    conn.commit()
    return [dict(r) for r in best.values()]


def selected_lr(conn: sqlite3.Connection, model: str, strategy: str, dataset: str) -> float | None:
    r = conn.execute("SELECT learning_rate FROM training_runs WHERE model=? AND strategy=? AND dataset=? AND selected=1",
                     (model, strategy, dataset)).fetchone()
    return float(r[0]) if r else None


def mock_training_row(plan: dict[str, Any], val_metric: float | None = None) -> dict[str, Any]:
    """Simulated row for offline pipeline tests only (is_mock=1)."""
    factor = {"fft": 3.0, "lora": 2.0, "qlora": 1.0}.get(plan["strategy"], 1.0)
    return {"model": plan["model"], "strategy": plan["strategy"], "dataset": plan["dataset"], "seed": plan["seed"],
            "learning_rate": plan["learning_rate"], "selected": 0,
            "val_metric": val_metric if val_metric is not None else round(0.4 + plan["learning_rate"] * 100, 4),
            "n_train_examples": 0, "epochs": plan["epochs"], "peak_vram_mb": 1000.0 * factor + plan["seed"] % 7,
            "duration_s": 60.0 * factor, "energy_j": 9000.0 * factor + plan["seed"], "avg_power_w": 150.0,
            "trainable_params": 1, "total_params": 1, "final_loss": 1.0,
            "optimizer": precision_recipe(plan, True)["label"],
            "grad_checkpointing": int(plan["gradient_checkpointing"]), "is_mock": 1}


def training_prompt(dataset: str, item: dict, plan: dict, count: Any = None) -> Any:
    """Same context mode, budget and template overhead as evaluation, so truncation is identical."""
    mode = "full" if dataset == "finqa" else "closed"
    return build_prompt(dataset, item, mode, None, plan["context_tokens"], plan["max_new_tokens"], count,
                        plan.get("prompt_overhead_tokens", 64))


def _tokenize_example(tok: Any, dataset: str, item: dict, plan: dict) -> dict[str, list[int]]:
    prompt = training_prompt(dataset, item, plan, lambda t: len(tok(t, add_special_tokens=False)["input_ids"]))
    prefix = render(prompt.messages, tok, add_generation_prompt=True, template_kwargs=plan["chat_template_kwargs"])
    full = render(prompt.messages + [{"role": "assistant", "content": target_text(dataset, item)}], tok,
                  add_generation_prompt=False, template_kwargs=plan["chat_template_kwargs"])
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][: plan["max_seq_len"]]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels[: len(f_ids)]}


def _validate(model: Any, tok: Any, dataset: str, items: list[dict], plan: dict) -> float:
    import torch

    from .scoring import score

    model.eval()
    correct = 0
    for it in items:
        prompt = training_prompt(dataset, it, plan, lambda t: len(tok(t, add_special_tokens=False)["input_ids"]))
        text = render(prompt.messages, tok, template_kwargs=plan["chat_template_kwargs"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=plan["max_new_tokens"], do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += int(score(dataset, reply, it).correct)
    return correct / len(items) if items else float("nan")


@dataclass
class TrainLimits:
    """Caps for feasibility benchmarks; a capped run is never written to training_runs."""

    max_steps: int | None = None
    time_limit_s: float | None = None
    max_train_examples: int | None = None
    skip_validation: bool = False
    skip_save: bool = False

    @property
    def capped(self) -> bool:
        return bool(self.max_steps or self.time_limit_s or self.max_train_examples)


def subsample(items: list[dict], n: int | None, seed: int = 42) -> list[dict]:
    """Seeded subset that is the same for every strategy, LR and training seed (fixed seed, file order kept)."""
    if not n or n >= len(items):
        return items
    keep = set(random.Random(seed).sample(range(len(items)), int(n)))
    return [it for i, it in enumerate(items) if i in keep]


class TokenCounter:
    """Wraps a data collator and counts real (attention-mask) and padded tokens of every micro-batch."""

    def __init__(self, collator: Any):
        self.collator = collator
        self.tokens = 0
        self.padded = 0

    def __call__(self, features: list[dict]) -> Any:
        batch = self.collator(features)
        mask = batch["attention_mask"]
        self.tokens += int(mask.sum())
        self.padded += int(mask.numel())
        return batch


def peak_rss_mb() -> float:
    """Peak resident set size of this process (Linux reports ru_maxrss in KiB)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_environment(threads: int | None) -> dict[str, Any]:
    from .env import cpu_model

    cpuset = None
    for p in ("/sys/fs/cgroup/cpuset/cpuset.effective_cpus", "/sys/fs/cgroup/cpuset/cpuset.cpus",
              "/sys/fs/cgroup/cpuset.cpus.effective"):
        try:
            cpuset = Path(p).read_text().strip()
            break
        except OSError:
            continue
    return {"device": f"cpu: {cpu_model()}", "cpu_threads": threads, "cpuset": cpuset,
            "cpu_bf16": cpu_bf16(), "gpu": None}


def _make_callback(timer: dict[str, Any], counter: TokenCounter, limits: TrainLimits) -> Any:
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            timer["steps"].append((time.perf_counter(), counter.tokens, counter.padded))
            if limits.time_limit_s and time.perf_counter() - timer["t0"] > limits.time_limit_s and state.global_step >= 3:
                control.should_training_stop = True
            return control

    return StepTimer()


def throughput(steps: list[tuple[float, int, int]]) -> dict[str, Any]:
    """Tokens/s over optimiser steps 2..n, so the first step's warm-up (allocation, kernel selection) is excluded."""
    if len(steps) < 2:
        return {"tokens_per_s": None, "padded_tokens_per_s": None, "s_per_step": None, "timed_steps": 0}
    (t0, k0, p0), (t1, k1, p1) = steps[0], steps[-1]
    dt = t1 - t0
    return {"tokens_per_s": (k1 - k0) / dt, "padded_tokens_per_s": (p1 - p0) / dt,
            "s_per_step": dt / (len(steps) - 1), "timed_steps": len(steps) - 1}


def tokenized_cache_path(plan: dict[str, Any]) -> Path:
    """Tokenised-training-set cache shared by every strategy, LR and seed of one model and dataset.

    Truncating long FinQA contexts re-tokenises them (about 0.3 s per example on CPU); the key covers every input."""
    import hashlib
    import json

    from .prompts import PROMPT_VERSION

    st = Path(plan["train_file"]).stat()
    ident = {k: plan.get(k) for k in ("hf_id", "dataset", "context_tokens", "max_new_tokens", "prompt_overhead_tokens",
                                      "max_seq_len", "chat_template_kwargs", "max_train_examples")}
    ident.update(train_file=str(plan["train_file"]), size=st.st_size, mtime=int(st.st_mtime), prompt=PROMPT_VERSION)
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return Path(plan["output_dir"]).parents[1] / "tokenized" / f"{plan['dataset']}-{key}.jsonl"


def tokenized_train_set(tok: Any, plan: dict[str, Any], items: list[dict]) -> list[dict[str, list[int]]]:
    import json
    import os

    path = tokenized_cache_path(plan)
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) == len(items):
            log.info("tokenised training set from cache %s", path)
            return rows
    rows = [_tokenize_example(tok, plan["dataset"], it, plan) for it in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(part, path)
    return rows


def run_training(plan: dict[str, Any], conn: sqlite3.Connection | None, nvml_index: int = 0,
                 val_limit: int | None = None, device: str = "auto", threads: int | None = None,
                 limits: TrainLimits | None = None) -> int | dict[str, Any]:
    """Train one model. Full runs write a training_runs row and return its id; capped benchmark runs
    (``limits``) return a metrics dict and write nothing. Needs the [gpu] extra (CUDA) or the CPU training image."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments, set_seed)

    from .telemetry.base import Sampler
    from .telemetry.cgroup import CgroupSource, read_cgroup
    from .telemetry.nvml import NvmlSource

    limits = limits or TrainLimits()
    dev = resolve_device(device, torch)
    on_cpu = dev == "cpu"
    if on_cpu and threads:
        torch.set_num_threads(int(threads))
    set_seed(plan["seed"])
    tok = AutoTokenizer.from_pretrained(plan["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    recipe = precision_recipe(plan, cpu_bf16() if on_cpu else native_bf16(torch), dev)
    kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, recipe["load_dtype"])}
    if plan["strategy"] == "qlora":
        from transformers import BitsAndBytesConfig

        q = plan["quantization"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(q.get("double_quant", True)),
            bnb_4bit_compute_dtype=getattr(torch, recipe["bnb_compute_dtype"]))
    cg = CgroupSource() if on_cpu else None
    cg = cg if cg is not None and cg.available() else None
    cpu0 = cg.cpu_s() if cg else None
    wall0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(plan["hf_id"], device_map={"": "cpu" if on_cpu else 0}, **kwargs)
    if plan["strategy"] in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if plan["strategy"] == "qlora" and plan["qlora_upcast_non_quantized"]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=plan["gradient_checkpointing"])
        lc = plan["lora"]
        model = get_peft_model(model, LoraConfig(r=int(lc["r"]), lora_alpha=int(lc["alpha"]),
                                                 lora_dropout=float(lc["dropout"]),
                                                 target_modules=lc.get("target_modules", "all-linear"),
                                                 task_type="CAUSAL_LM"))
    if plan["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train_items = subsample(read_jsonl(plan["train_file"]), plan.get("max_train_examples"))
    if limits.max_train_examples:
        train_items = train_items[: limits.max_train_examples]
    val_items = read_jsonl(plan["val_file"])[: val_limit] if val_limit else read_jsonl(plan["val_file"])
    if limits.capped:
        train_ds = [_tokenize_example(tok, plan["dataset"], it, plan) for it in train_items]
    else:
        train_ds = tokenized_train_set(tok, plan, train_items)
    args = TrainingArguments(
        output_dir=plan["output_dir"], num_train_epochs=plan["epochs"], max_steps=limits.max_steps or -1,
        per_device_train_batch_size=plan["per_device_batch_size"],
        gradient_accumulation_steps=plan["gradient_accumulation_steps"], learning_rate=plan["learning_rate"],
        lr_scheduler_type=plan["lr_scheduler"], warmup_ratio=plan["warmup_ratio"], weight_decay=plan["weight_decay"],
        optim=plan["optimizer"], bf16=recipe["bf16"], fp16=recipe["fp16"], logging_steps=plan["logging_steps"],
        save_strategy="no", report_to=[], seed=plan["seed"], gradient_checkpointing=False, use_cpu=on_cpu,
        dataloader_num_workers=0)
    counter = TokenCounter(DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    timer: dict[str, Any] = {"t0": time.perf_counter(), "steps": []}
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=counter,
                      callbacks=[_make_callback(timer, counter, limits)])

    nv = None if on_cpu else NvmlSource(nvml_index)
    sampler = Sampler(nv, 0.1, keep_s=1e9) if nv is not None and nv.available() else None
    e0 = nv.energy_j() if sampler else None
    if not on_cpu:
        torch.cuda.reset_peak_memory_stats()
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    timer["t0"] = t0
    train_cpu0 = cg.cpu_s() if cg else None
    result = trainer.train()
    duration = time.perf_counter() - t0
    train_cpu = cg.cpu_s() - train_cpu0 if cg else None
    if sampler:
        sampler.stop()
    e1 = nv.energy_j() if sampler else None
    trace = sampler.samples if sampler else []
    energy = (e1 - e0) if e0 is not None and e1 is not None else None
    if on_cpu:
        peak_vram, peak_nvml = None, None
        peak_mem, mem_kind = peak_rss_mb(), "RSS-peak"
        cg_reading = read_cgroup(cg.paths) if cg else None
        cg_peak = cg_reading.mem_peak_bytes / (1024 * 1024) if cg_reading and cg_reading.mem_peak_bytes else None
        log.info("peak RAM: RSS=%.0f MB, cgroup peak=%s MB (incl. page cache)", peak_mem, cg_peak)
    else:
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_nvml = max((r.vram_mb for r in trace if r.vram_mb is not None), default=None)
        peak_mem, mem_kind, cg_peak = None, None, None
        log.info("peak VRAM torch=%.0f MB, NVML=%s MB", peak_vram, peak_nvml)

    metrics = {"model": plan["model"], "strategy": plan["strategy"], "dataset": plan["dataset"],
               "device": dev, "threads": threads, "recipe": recipe["label"], "train_s": duration,
               "train_cpu_s": train_cpu, "optimizer_steps": int(result.global_step),
               "sequences": int(result.global_step) * plan["effective_batch_size"],
               "peak_rss_mb": peak_mem, "cgroup_peak_mb": cg_peak, "peak_vram_mb": peak_vram,
               "final_loss": float(result.training_loss), "trainable_params": trainable, "total_params": total,
               "tokens": counter.tokens, "padded_tokens": counter.padded, **throughput(timer["steps"])}
    if limits.capped:
        if not limits.skip_validation and val_items:
            v0 = time.perf_counter()
            metrics["val_accuracy_subset"] = _validate(model, tok, plan["dataset"], val_items, plan)
            metrics["val_s_per_item"] = (time.perf_counter() - v0) / len(val_items)
        metrics["wall_s_total"] = time.perf_counter() - wall0
        metrics["cpu_s_total"] = cg.cpu_s() - cpu0 if cg else None
        if nv is not None:
            nv.close()
        return metrics

    val = _validate(model, tok, plan["dataset"], val_items, plan) if plan["val_metric"] == "accuracy" else None
    Path(plan["output_dir"]).mkdir(parents=True, exist_ok=True)
    if not limits.skip_save:
        model.save_pretrained(plan["output_dir"])
        tok.save_pretrained(plan["output_dir"])
    row = {"model": plan["model"], "strategy": plan["strategy"], "dataset": plan["dataset"], "seed": plan["seed"],
           "learning_rate": plan["learning_rate"], "selected": 0, "val_metric": val,
           "n_train_examples": len(train_ds), "epochs": plan["epochs"], "peak_vram_mb": peak_vram,
           "duration_s": duration, "energy_j": energy, "avg_power_w": energy / duration if energy else None,
           "trainable_params": trainable, "total_params": total, "final_loss": float(result.training_loss),
           "optimizer": recipe["label"], "grad_checkpointing": int(plan["gradient_checkpointing"]), "is_mock": 0,
           "peak_mem_mb": peak_mem, "mem_kind": mem_kind, "device": dev, "cpu_s": train_cpu}
    if nv is not None:
        nv.close()
    train_id = db.insert_training_run(conn, row)
    env: dict[str, Any] = {f"train/{train_id}/energy_source": "nvml-counter" if energy is not None else "none"}
    if on_cpu:
        env.update({f"train/{train_id}/{k}": v for k, v in cpu_environment(threads).items()})
        env[f"train/{train_id}/cgroup_peak_mb"] = cg_peak
        env[f"train/{train_id}/wall_s_total"] = time.perf_counter() - wall0
        env[f"train/{train_id}/cpu_s_total"] = cg.cpu_s() - cpu0 if cg else None
        if train_cpu is not None:
            db.upsert_cpu_usage(conn, {"scope": "train", "ref_id": str(train_id), "device": dev,
                                       "wall_s": duration, "cpu_s": train_cpu, "peak_mem_mb": peak_mem,
                                       "mem_kind": mem_kind, "source": "container-cgroup", "coverage": "trainer.train()"})
    else:
        env[f"train/{train_id}/peak_vram_nvml_mb"] = peak_nvml
    db.set_env(conn, env)
    db.insert_telemetry(conn, "train", str(train_id), [r.as_dict() for r in trace])
    return train_id


def token_stats(plan: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Token counts of the training set exactly as training tokenises it (template, truncation, max_seq_len)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(plan["hf_id"])
    items = subsample(read_jsonl(plan["train_file"]), plan.get("max_train_examples"))
    items = items[: limit] if limit else items
    lens = sorted(len(_tokenize_example(tok, plan["dataset"], it, plan)["input_ids"]) for it in items)
    n = len(lens)
    pick = (lambda q: lens[min(n - 1, int(q * n))]) if n else (lambda q: None)
    return {"model": plan["model"], "dataset": plan["dataset"], "n_examples": n, "total_tokens": sum(lens),
            "mean_tokens": sum(lens) / n if n else None, "p50": pick(0.5), "p95": pick(0.95),
            "max": lens[-1] if n else None}
