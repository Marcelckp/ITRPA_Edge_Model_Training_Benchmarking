"""Load the experimental matrix and expand it into concrete, identifiable configurations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs"
DEFAULT_MATRIX = DEFAULT_CONFIG_DIR / "matrix.yaml"
DEFAULT_DEVICES_DIR = DEFAULT_CONFIG_DIR / "devices"
DEFAULT_TARIFF = DEFAULT_CONFIG_DIR / "tariff.yaml"

KNOWN_STRATEGIES = {"base", "fft", "lora", "qlora", "rag", "celora"}
KNOWN_DATASETS = {"pubmedqa", "finqa", "arc_easy"}
KNOWN_ENGINES = {"vllm", "llamacpp", "ollama"}
KNOWN_PRECISIONS = {"bf16", "F16", "Q8_0", "Q4_K_M"}
RETENTION_PREFIX = "retention:"

# Fields that define a configuration's identity. model_path is excluded because it differs per machine.
ID_FIELDS = (
    "model", "strategy", "precision", "engine", "tier", "device", "dataset",
    "context_mode", "k", "seed", "temperature", "max_new_tokens",
)


class ConfigError(ValueError):
    """Raised when a matrix or device file is invalid."""


@dataclass
class ExpConfig:
    model: str
    strategy: str
    precision: str
    engine: str
    tier: str
    device: str
    dataset: str
    context_mode: str
    k: int
    seed: int
    temperature: float
    max_new_tokens: int
    model_path: str = ""
    is_mock: int = 0
    plan: str = ""
    train_dataset: str = ""
    hf_id: str = ""
    n_items: int | None = None
    config_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.config_id = make_config_id(self)

    def db_row(self) -> dict[str, Any]:
        keys = ("config_id",) + ID_FIELDS + ("model_path", "is_mock")
        return {k: getattr(self, k) for k in keys}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_config_id(cfg: ExpConfig | dict[str, Any]) -> str:
    d = cfg.as_dict() if isinstance(cfg, ExpConfig) else dict(cfg)
    ident = {k: d[k] for k in ID_FIELDS}
    ident["temperature"] = float(ident["temperature"])
    ident["k"] = int(ident["k"])
    if int(d.get("is_mock", 0)):
        ident["is_mock"] = 1
    blob = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_matrix(path: str | os.PathLike = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = load_yaml(path)
    validate_matrix(matrix)
    return matrix


def validate_matrix(m: dict[str, Any]) -> None:
    for key in ("models", "strategies", "datasets", "plans", "max_new_tokens"):
        if key not in m:
            raise ConfigError(f"matrix: missing '{key}'")
    if not m["models"]:
        raise ConfigError("matrix: no models")
    ids = [x["id"] for x in m["models"]]
    if len(ids) != len(set(ids)):
        raise ConfigError("matrix: duplicate model ids")
    unknown = set(m["strategies"]) - KNOWN_STRATEGIES
    if unknown:
        raise ConfigError(f"matrix: unknown strategies {sorted(unknown)}")
    unknown = set(m["datasets"]) - KNOWN_DATASETS
    if unknown:
        raise ConfigError(f"matrix: unknown datasets {sorted(unknown)}")
    for ds in m["datasets"]:
        if ds not in m["max_new_tokens"]:
            raise ConfigError(f"matrix: max_new_tokens missing for {ds}")
    for k in m.get("rag_k", []):
        if not isinstance(k, int) or k <= 0:
            raise ConfigError(f"matrix: rag_k values must be positive ints, got {k!r}")
    for name, plan in m["plans"].items():
        for key in ("tier", "engine", "devices", "precisions", "datasets"):
            if key not in plan:
                raise ConfigError(f"plan {name}: missing '{key}'")
        if plan["engine"] not in KNOWN_ENGINES:
            raise ConfigError(f"plan {name}: unknown engine {plan['engine']}")
        bad = set(plan["precisions"]) - KNOWN_PRECISIONS
        if bad:
            raise ConfigError(f"plan {name}: unknown precisions {sorted(bad)}")
        bad = set(plan["datasets"]) - set(m["datasets"])
        if bad:
            raise ConfigError(f"plan {name}: datasets not defined {sorted(bad)}")
        if plan.get("subset") is not None and int(plan["subset"]) <= 0:
            raise ConfigError(f"plan {name}: subset must be positive or null")
        if plan["engine"] == "vllm" and set(plan["precisions"]) != {"bf16"}:
            raise ConfigError(f"plan {name}: vllm plans use bf16 HF weights only (DESIGN.md 3)")


def load_device(name_or_path: str, devices_dir: str | os.PathLike = DEFAULT_DEVICES_DIR) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.suffix:
        p = Path(devices_dir) / f"{name_or_path}.yaml"
    dev = load_yaml(p)
    for key in ("name", "tier", "backends"):
        if key not in dev:
            raise ConfigError(f"{p}: missing '{key}'")
    for eng, b in dev["backends"].items():
        url = str(b.get("base_url", ""))
        if url and not _is_loopback_url(url):
            raise ConfigError(f"{p}: backend {eng} base_url must be loopback, got {url}")
    return dev


def load_devices(names: Iterable[str], devices_dir: str | os.PathLike = DEFAULT_DEVICES_DIR) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for n in names:
        try:
            out[n] = load_device(n, devices_dir)
        except FileNotFoundError:
            log.debug("device file for %s not found; using matrix defaults", n)
    return out


def _is_loopback_url(url: str) -> bool:
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def load_tariff(path: str | os.PathLike = DEFAULT_TARIFF, tariff: float | None = None,
                grid_factor: float | None = None) -> dict[str, Any]:
    data = load_yaml(path)
    t = data["tariff_zar_per_kwh"]
    g = data["grid_kgco2e_per_kwh"]
    out = {
        "tariff_zar_per_kwh": float(t["value"]),
        "tariff_source": str(t.get("source", "")),
        "grid_kgco2e_per_kwh": float(g["value"]),
        "grid_source": str(g.get("source", "")),
    }
    if tariff is not None:
        out["tariff_zar_per_kwh"], out["tariff_source"] = float(tariff), "CLI override"
    if grid_factor is not None:
        out["grid_kgco2e_per_kwh"], out["grid_source"] = float(grid_factor), "CLI override"
    out["is_placeholder"] = any(s.upper().startswith("TODO") for s in (out["tariff_source"], out["grid_source"]))
    return out


def resolve_model_path(matrix: dict, device: dict | None, model: dict, strategy: str,
                       precision: str, train_dataset: str) -> str:
    paths = matrix.get("paths", {})
    models_dir = (device or {}).get("paths", {}).get("models_dir") or paths.get("models_dir", "models")
    trained = matrix["strategies"][strategy].get("trained", False)
    fmt = {"models_dir": models_dir, "model": model["id"], "strategy": strategy,
           "train_dataset": train_dataset, "precision": precision}
    if precision == "bf16":
        if not trained:
            return model.get("hf_id", model["id"])
        tmpl = paths.get("hf_merged", "{models_dir}/{model}/{strategy}-{train_dataset}/merged")
    else:
        key = "gguf" if trained else "base_gguf"
        default = ("{models_dir}/{model}/{strategy}-{train_dataset}/{model}-{strategy}-{train_dataset}-{precision}.gguf"
                   if trained else "{models_dir}/{model}/base/{model}-base-{precision}.gguf")
        tmpl = paths.get(key, default)
    return os.path.expanduser(tmpl.format(**fmt))


def _matches(cfg: ExpConfig, rule: dict[str, Any]) -> bool:
    for key, val in rule.items():
        have = getattr(cfg, key, None)
        vals = val if isinstance(val, list) else [val]
        if have not in vals and str(have) not in [str(v) for v in vals]:
            return False
    return True


def _context_variants(matrix: dict, ds_name: str, strategy: str,
                      plan: dict | None = None) -> list[tuple[str, int, str]]:
    """Return (context_mode, k, train_dataset) tuples that apply to this dataset/strategy pair."""
    ds = matrix["datasets"][ds_name]
    sdef = matrix["strategies"][strategy]
    override = (plan or {}).get("context_mode")
    if override:
        if sdef.get("retrieval") or ds.get("role") == "retention":
            return []
        return [(override, 0, ds_name if sdef.get("trained") else "")]
    if ds.get("role") == "retention":
        if sdef.get("retrieval"):
            return []
        if not sdef.get("trained"):
            return [(f"{RETENTION_PREFIX}none", 0, "")]
        return [(f"{RETENTION_PREFIX}{d}", 0, d) for d in ds.get("retention_after", [])]
    train_ds = ds_name if sdef.get("trained") else ""
    if sdef.get("retrieval"):
        mode = ds.get("rag_context_mode", "retrieved")
        return [(mode, int(k), "") for k in matrix.get("rag_k", [])]
    return [(ds.get("context_mode", "closed"), 0, train_ds)]


def enabled_strategies(matrix: dict) -> list[str]:
    return [s for s, d in matrix["strategies"].items() if d.get("enabled", True)]


def expand(matrix: dict, plans: Iterable[str] | None = None, devices: dict[str, dict] | None = None,
           only_devices: Iterable[str] | None = None, is_mock: bool = False,
           filters: dict[str, Any] | None = None) -> list[ExpConfig]:
    """Expand the matrix into concrete configurations, skipping combinations that do not apply."""
    plan_names = list(plans) if plans else list(matrix["plans"])
    devices = devices or {}
    only = set(only_devices) if only_devices else None
    seed = int(matrix.get("seed", 42))
    temp = float(matrix.get("temperature", 0.0))
    out: list[ExpConfig] = []
    seen: set[str] = set()
    for pname in plan_names:
        if pname not in matrix["plans"]:
            raise ConfigError(f"unknown plan {pname}")
        plan = matrix["plans"][pname]
        for dev_name in plan["devices"]:
            if only and dev_name not in only:
                continue
            dev = devices.get(dev_name)
            strategies = [s for s in enabled_strategies(matrix)
                          if not plan.get("strategies") or s in plan["strategies"]]
            for model in matrix["models"]:
                for strategy in strategies:
                    for ds_name in plan["datasets"]:
                        ds = matrix["datasets"][ds_name]
                        for mode, k, train_ds in _context_variants(matrix, ds_name, strategy, plan):
                            for prec in plan["precisions"]:
                                n = ds.get("n_test")
                                if plan.get("subset") is not None:
                                    n = min(int(plan["subset"]), int(n)) if n else int(plan["subset"])
                                cfg = ExpConfig(
                                    model=model["id"], strategy=strategy, precision=prec,
                                    engine=plan["engine"], tier=plan["tier"], device=dev_name,
                                    dataset=ds_name, context_mode=mode, k=k, seed=seed,
                                    temperature=temp, max_new_tokens=int(matrix["max_new_tokens"][ds_name]),
                                    model_path=resolve_model_path(matrix, dev, model, strategy, prec, train_ds),
                                    is_mock=int(is_mock), plan=pname, train_dataset=train_ds,
                                    hf_id=model.get("hf_id", ""), n_items=n,
                                )
                                if any(_matches(cfg, r) for r in plan.get("exclude", []) or []):
                                    continue
                                if filters and not _matches(cfg, filters):
                                    continue
                                if cfg.config_id in seen:
                                    continue
                                seen.add(cfg.config_id)
                                out.append(cfg)
    log.info("expanded %d configurations from plans %s", len(out), plan_names)
    return out


def count_runs(configs: Iterable[ExpConfig]) -> dict[str, Any]:
    """Count configurations and query executions, overall and per plan/device."""
    by: dict[str, dict[str, int]] = {}
    total_cfg = total_q = 0
    for c in configs:
        key = f"{c.plan}/{c.device}"
        b = by.setdefault(key, {"configs": 0, "queries": 0})
        b["configs"] += 1
        b["queries"] += int(c.n_items or 0)
        total_cfg += 1
        total_q += int(c.n_items or 0)
    return {"configs": total_cfg, "queries": total_q, "by_plan_device": by}


def training_plan(matrix: dict, phase: str = "selection") -> list[dict[str, Any]]:
    """Training jobs: 'selection' = every LR in the grid at the selection seed; 'variability' = extra
    seeds at the (later) selected LR, recorded with lr=None until selection has happened."""
    jobs = []
    tr = matrix.get("training", {})
    for model in matrix["models"]:
        for s in enabled_strategies(matrix):
            if not matrix["strategies"][s].get("trained"):
                continue
            for ds in tr.get("datasets", {}):
                base = {"model": model["id"], "hf_id": model.get("hf_id"), "strategy": s, "dataset": ds}
                if phase == "selection":
                    for lr in tr.get("lr_grid", {}).get(s, []):
                        jobs.append({**base, "seed": int(tr.get("selection_seed", 42)), "learning_rate": float(lr)})
                elif phase == "variability":
                    for seed in tr.get("variability_seeds", []):
                        jobs.append({**base, "seed": int(seed), "learning_rate": None})
                else:
                    raise ConfigError(f"unknown training phase {phase}")
    return jobs


def adapted_model_count(matrix: dict) -> int:
    trained = [s for s in enabled_strategies(matrix) if matrix["strategies"][s].get("trained")]
    return len(matrix["models"]) * len(trained) * len(matrix.get("training", {}).get("datasets", {}))


def trained_model_dir(matrix: dict, model: str, strategy: str, dataset: str, lr: float, seed: int,
                      models_dir: str | None = None) -> str:
    paths = matrix.get("paths", {})
    tmpl = paths.get("hf_trained", "{models_dir}/{model}/{strategy}-{train_dataset}/lr{lr}-s{seed}")
    return os.path.expanduser(tmpl.format(models_dir=models_dir or paths.get("models_dir", "models"),
                                          model=model, strategy=strategy, train_dataset=dataset,
                                          lr=f"{lr:g}", seed=seed))


def merged_model_dir(matrix: dict, model: str, strategy: str, dataset: str, models_dir: str | None = None) -> str:
    paths = matrix.get("paths", {})
    tmpl = paths.get("hf_merged", "{models_dir}/{model}/{strategy}-{train_dataset}/merged")
    return os.path.expanduser(tmpl.format(models_dir=models_dir or paths.get("models_dir", "models"),
                                          model=model, strategy=strategy, train_dataset=dataset))


def plan_of(row: dict[str, Any], matrix: dict) -> str:
    """Recover the plan a stored config came from (configs has no plan column)."""
    for name, p in matrix["plans"].items():
        if row.get("engine") != p["engine"] or row.get("tier") != p["tier"]:
            continue
        if row.get("device") not in p["devices"] or row.get("dataset") not in p["datasets"]:
            continue
        if row.get("precision") not in p["precisions"]:
            continue
        if p.get("strategies") and row.get("strategy") not in p["strategies"]:
            continue
        if p.get("context_mode") and row.get("context_mode") != p["context_mode"]:
            continue
        if not p.get("context_mode") and row.get("context_mode") == "gold":
            continue
        return name
    return ""


def parse_filters(items: Iterable[str] | None) -> dict[str, Any]:
    """Parse CLI filters like ``model=qwen2.5-0.5b`` or ``k=3,5``."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"filter must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        vals: list[Any] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
        out[k.strip()] = vals
    return out
