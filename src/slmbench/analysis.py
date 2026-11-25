"""Analysis pipeline: tidy tables, hypothesis tests with the DESIGN.md v2 decision rules, cost and parity."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import MOCK_WATERMARK, db
from . import stats as S
from .config import RETENTION_PREFIX, plan_of
from .scoring import macro_f1, per_class_recall

log = logging.getLogger(__name__)

class MockDataRefused(RuntimeError):
    pass


def metered_source(source: Any) -> bool:
    """PMIC rails are a partial lower bound (DESIGN.md 3.5), so they never count as metered energy."""
    return isinstance(source, str) and bool(source) and not source.startswith("pmic")


def planned_items(cfg: dict[str, Any], matrix: dict) -> int | None:
    """Items a full block of this config runs: the plan's subset, capped at the dataset's test size."""
    plan = matrix.get("plans", {}).get(cfg.get("plan") or "", {})
    n = matrix.get("datasets", {}).get(cfg.get("dataset"), {}).get("n_test")
    sub = plan.get("subset")
    if sub is not None:
        return min(int(sub), int(n)) if n else int(sub)
    return int(n) if n else None


def _settings(matrix: dict) -> dict[str, Any]:
    a = matrix.get("analysis", {})
    plans = matrix.get("plans", {})
    edge_plan = a.get("edge_plan", "edge")
    strategies = matrix.get("strategies") or {s: {"trained": True} for s in ("fft", "lora", "qlora")}
    trained = [k for k, v in strategies.items() if v.get("trained") and v.get("enabled", True)]
    return {"precisions": list(a.get("precisions") or plans.get(edge_plan, {}).get("precisions")
                               or ["F16", "Q8_0", "Q4_K_M"]),
            "k_levels": [int(k) for k in matrix.get("rag_k", [3, 5, 10])],
            "ft": [x for x in ("fft", "lora", "qlora") if x in trained],
            "edge_plan": edge_plan,
            "alpha": float(a.get("alpha", 0.05)), "n_boot": int(a.get("bootstrap_resamples", 10000)),
            "seed": int(a.get("bootstrap_seed", 42)), "delta": float(a.get("noninferiority_delta", 0.05)),
            "min_disc": int(a.get("floor_min_discordant", 10)), "min_acc": float(a.get("floor_min_accuracy", 0.05)),
            "parity": float(a.get("parity_threshold", 0.98)), "acc_plan": a.get("accuracy_plan", "accuracy"),
            "ret_plan": a.get("retention_plan", "retention"), "sq2_k": int(a.get("rag_k_for_sq2", 5)),
            "selection_seed": int(matrix.get("training", {}).get("selection_seed", 42))}


def load_frames(conn: sqlite3.Connection, matrix: dict, allow_mock: bool = False) -> dict[str, pd.DataFrame]:
    where = "" if allow_mock else " WHERE is_mock=0"
    configs = pd.read_sql_query("SELECT * FROM configs" + where, conn)
    runs = pd.read_sql_query("SELECT * FROM runs" + where, conn)
    blocks = pd.read_sql_query("SELECT * FROM energy_blocks" + where, conn)
    training = pd.read_sql_query("SELECT * FROM training_runs" + where, conn)
    if not configs.empty:
        configs["plan"] = [plan_of(r, matrix) for r in configs.to_dict("records")]
        configs["variant"] = [s if s != "rag" else f"rag-k{int(k)}" for s, k in zip(configs.strategy, configs.k)]
        configs["train_domain"] = [cm[len(RETENTION_PREFIX):] if str(cm).startswith(RETENTION_PREFIX) else ""
                                   for cm in configs.context_mode]
    df = runs.drop(columns=["is_mock"]).merge(configs, on="config_id", how="left") if not runs.empty else runs
    if not df.empty:
        idle = blocks.set_index("block_id")["idle_power_w"] if not blocks.empty else pd.Series(dtype=float)
        df["idle_power_w"] = df["block_id"].map(idle)
        df["net_energy_j"] = df["energy_j"] - df["idle_power_w"] * df["e2e_latency_s"]
        df["ttft_primary_s"] = np.where(df["server_prompt_ms"].notna(), df["server_prompt_ms"] / 1000.0, df["ttft_s"])
        tp = df["server_predicted_ms"] / 1000.0 / df["output_tokens"].replace(0, np.nan)
        df["tpot_primary_s"] = np.where(tp.notna(), tp, df["tpot_s"])
    if not configs.empty:
        configs["planned_n"] = [planned_items(r, matrix) for r in configs.to_dict("records")]
    if not blocks.empty and not configs.empty:
        blocks = blocks.drop(columns=["is_mock"]).merge(configs, on=["config_id", "device"], how="left")
        blocks["net_energy_j"] = blocks["energy_j"] - blocks["idle_energy_j"]
        blocks["gross_j_per_query"] = blocks["energy_j"] / blocks["n_queries"].replace(0, np.nan)
        blocks["net_j_per_query"] = blocks["net_energy_j"] / blocks["n_queries"].replace(0, np.nan)
    return {"runs": df, "configs": configs, "blocks": blocks, "training": training}


def _expand_retention_base(df: pd.DataFrame, domains: Iterable[str]) -> pd.DataFrame:
    """Base ARC rows are shared by every training domain's retention family."""
    base = df[(df.dataset == "arc_easy") & (df.train_domain == "")]
    other = df[~((df.dataset == "arc_easy") & (df.train_domain == ""))]
    copies = [base.assign(train_domain=d) for d in domains]
    return pd.concat([other, *copies], ignore_index=True) if copies else df


def config_summary(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    rows = []
    for cid, g in df.groupby("config_id", sort=True):
        first = g.iloc[0]
        ok = g[g.error.isna()]
        boot = S.bootstrap_ci(ok.correct.astype(float).tolist(), n_boot=s["n_boot"], seed=s["seed"])
        row = {k: first[k] for k in ("config_id", "plan", "model", "strategy", "variant", "precision", "engine",
                                     "tier", "device", "dataset", "context_mode", "k", "train_domain")}
        row.update(is_mock=int(first.get("is_mock", 0) or 0), n=len(g), n_errors=int(g.error.notna().sum()),
                   parse_fail_rate=float(1 - ok.parse_ok.mean()) if len(ok) else None,
                   accuracy=boot["estimate"], acc_lo=boot["lo"], acc_hi=boot["hi"],
                   truncated=int(ok.truncated.fillna(0).sum()),
                   retrieval_hit_rate=float(ok.retrieval_hit.mean()) if ok.retrieval_hit.notna().any() else None,
                   throttled=int(ok.throttled.fillna(0).sum()))
        if first["dataset"] == "pubmedqa" and len(ok):
            row["macro_f1"] = macro_f1(ok.gold_answer.tolist(), ok.parsed_answer.tolist())
            for lab, rec in per_class_recall(ok.gold_answer.tolist(), ok.parsed_answer.tolist()).items():
                row[f"recall_{lab}"] = rec
        for metric in ("ttft_primary_s", "tpot_primary_s", "e2e_latency_s", "peak_mem_mb", "net_energy_j", "energy_j"):
            d = S.describe(ok[metric].tolist())
            for stat_name in ("mean", "sd", "median", "iqr"):
                row[f"{metric}_{stat_name}"] = d[stat_name]
        row["mem_kind"] = ok.mem_kind.dropna().iloc[0] if ok.mem_kind.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, factor: str, levels: list[Any], metric: str) -> tuple[pd.DataFrame, int]:
    """Items x levels table with listwise removal of items that errored or are missing in any level."""
    sub = df[df[factor].astype(str).isin([str(x) for x in levels])].copy()
    sub["_lvl"] = sub[factor].astype(str)
    errored = set(sub.loc[sub.error.notna(), "query_id"])
    wide = sub[sub.error.isna()].pivot_table(index="query_id", columns="_lvl", values=metric, aggfunc="first")
    cols = [str(x) for x in levels if str(x) in wide.columns]
    wide = wide[cols] if cols else wide.iloc[:, :0]
    before = wide.shape[0] + len(errored - set(wide.index))
    wide = wide.drop(index=[q for q in errored if q in wide.index]).dropna()
    return wide, before - wide.shape[0]


def _fmt_group(keys: Iterable[str], values: Iterable[Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(keys, values))


def _decision_row(hyp: str, family: str, group: str, comparison: str, res: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"hypothesis": hyp, "family": family, "group": group, "comparison": comparison,
            "test": res.get("test") or res.get("method"), "n": res.get("n"), "statistic": res.get("statistic"),
            "p": res.get("p"), "p_holm": None, "effect": kw.pop("effect", None), "ci": kw.pop("ci", None),
            "decision": kw.pop("decision", None), "reason": res.get("reason"), **kw}


def _apply_holm(rows: list[dict[str, Any]], alpha: float) -> None:
    fams: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("holm", True):
            fams.setdefault(r["family"], []).append(i)
    for idx in fams.values():
        for i, p in zip(idx, S.holm([rows[i]["p"] for i in idx])):
            rows[i]["p_holm"] = p
    for r in rows:
        r.pop("holm", None)
        if r["decision"] is None:
            if r.get("reason"):
                r["decision"] = "not testable" if str(r["reason"]).startswith("not testable") else "no test"
            elif r["p_holm"] is None:
                r["decision"] = "no test"
            elif "direction_ok" in r:
                # Directional hypotheses need the adjusted p AND the predicted direction (DESIGN.md 2).
                r["decision"] = "supported" if r["p_holm"] < alpha and r["direction_ok"] else "not supported"
            else:
                r["decision"] = "reject H0" if r["p_holm"] < alpha else "fail to reject H0"
    _finalize_h3(rows, alpha)


def _finalize_h3(rows: list[dict[str, Any]], alpha: float) -> None:
    """H3 diminishing-returns verdict from Holm-adjusted p values; the floor rule is never overridden."""
    for r in rows:
        wil = r.pop("_h3_wilcoxon", None)
        if wil is None:
            continue
        ci_below = r.pop("_ci_below", False)
        if r.get("reason"):
            r["decision"] = "not testable"
            continue
        mc_ns = r["p_holm"] is None or r["p_holm"] >= alpha
        wil_sig = wil.get("p_holm") is not None and wil["p_holm"] < alpha and bool(wil.get("direction_ok"))
        r["decision"] = "supported" if (mc_ns and wil_sig) or ci_below else "not supported"


def h1a(training: pd.DataFrame, s: dict) -> tuple[list[dict], pd.DataFrame]:
    rows, summ = [], []
    if training.empty:
        return [{"hypothesis": "H1a", "family": "H1a", "group": "all", "comparison": "qlora < lora < fft",
                 "test": "descriptive ordering", "decision": "not testable", "reason": "no training runs",
                 "p": None, "p_holm": None, "holm": False}], pd.DataFrame()
    tr = training.copy()
    sel = tr[(tr.selected == 1) | (tr.seed != s["selection_seed"])]
    tr = sel if not sel.empty else tr
    for (ds, model, strat), g in tr.groupby(["dataset", "model", "strategy"]):
        on_gpu = "peak_mem_mb" not in g or g.peak_vram_mb.notna().any()
        mem = g.peak_vram_mb if on_gpu else g.peak_mem_mb
        d = S.describe(mem.tolist())
        summ.append({"dataset": ds, "model": model, "strategy": strat, "n_runs": d["n"], "peak_vram_mb_mean": d["mean"],
                     "peak_vram_cv": S.coefficient_of_variation(mem.tolist()),
                     "mem_kind": "VRAM" if on_gpu else str(g.mem_kind.iloc[0]),
                     "duration_s_mean": g.duration_s.mean(), "duration_cv": S.coefficient_of_variation(g.duration_s.tolist()),
                     "energy_j_mean": g.energy_j.mean(), "energy_cv": S.coefficient_of_variation(g.energy_j.dropna().tolist()),
                     "trainable_params": g.trainable_params.iloc[0], "total_params": g.total_params.iloc[0],
                     "is_mock": int(g.is_mock.max())})
    sdf = pd.DataFrame(summ)
    for ds, g in sdf.groupby("dataset"):
        per_model = []
        for model, gm in g.groupby("model"):
            means = gm.set_index("strategy").peak_vram_mb_mean
            vals = [means.get(x) for x in ("qlora", "lora", "fft")]
            per_model.append((model, S.ordered([None if v is None or pd.isna(v) else v for v in vals], "increasing")))
        if any(o is None for _, o in per_model):
            decision, reason = "not testable", "missing strategy means for some models"
        else:
            ok = all(o for _, o in per_model)
            decision, reason = ("supported" if ok else "not supported"), None
        tier = "gpu-workstation" if (g.get("mem_kind") is None or (g.mem_kind == "VRAM").all()) else "cpu"
        rows.append({"hypothesis": "H1a", "family": f"H1a/{ds}/{tier}", "group": f"dataset={ds}",
                     "comparison": f"qlora < lora < fft (peak {'VRAM' if tier != 'cpu' else 'RAM, ' + str(g.mem_kind.iloc[0])}) in all models", "test": "descriptive ordering",
                     "n": len(per_model), "statistic": None, "p": None, "p_holm": None,
                     "effect": "; ".join(f"{m}:{'holds' if o else 'fails'}" for m, o in per_model),
                     "ci": None, "decision": decision, "reason": reason, "holm": False})
    return rows, sdf


def h1b_memory(df: pd.DataFrame, s: dict) -> tuple[list[dict], list[dict]]:
    rows, post = [], []
    edge = df[df.tier == "edge"]
    for keys, g in edge.groupby(["device", "dataset", "model", "variant"]):
        device, ds, model, variant = keys
        precs = s["precisions"]
        wide, removed = _pivot(g, "precision", precs, "peak_mem_mb")
        res = S.friedman(wide.values) if wide.shape[1] == len(precs) else {"test": "friedman", "reason": "missing precisions"}
        med = [float(wide[c].median()) if c in wide else None for c in precs]
        consistent = S.ordered(med, "decreasing")
        fam = f"H1b-mem/{ds}/{device}"
        grp = _fmt_group(["model", "variant"], [model, variant])
        rows.append(_decision_row("H1b", fam, grp, " > ".join(precs) + " (peak memory)", res,
                                  effect=f"W={res.get('kendalls_w')}", direction_ok=bool(consistent), removed=removed,
                                  medians=med))
        if wide.shape[1] == len(precs) and not res.get("reason"):
            for r in S.posthoc_wilcoxon(wide.values, precs):
                post.append({"family": fam, "group": grp, "metric": "peak_mem_mb", **r})
    return rows, post


def block_energy(blocks: pd.DataFrame, s: dict) -> tuple[list[dict], pd.DataFrame]:
    rows, summ = [], []
    if blocks.empty:
        return rows, pd.DataFrame()
    blocks = blocks[blocks.energy_j.notna() & blocks.source.map(metered_source)]
    # A resumed or pilot block covers only part of the item set, so its per-query mean is not comparable.
    full = blocks.planned_n.isna() | (blocks.n_queries >= blocks.planned_n) if "planned_n" in blocks else \
        pd.Series(True, index=blocks.index)
    partial = blocks[~full]
    blocks = blocks[full]
    edge = blocks[blocks.tier == "edge"]
    for keys, g in blocks.groupby(["device", "dataset", "model", "variant", "precision"]):
        ci = S.bootstrap_ci(g.net_j_per_query.tolist(), n_boot=s["n_boot"], seed=s["seed"])
        n_part = int(((partial.device == keys[0]) & (partial.dataset == keys[1]) & (partial.model == keys[2])
                      & (partial.variant == keys[3]) & (partial.precision == keys[4])).sum()) if len(partial) else 0
        summ.append(dict(zip(["device", "dataset", "model", "variant", "precision"], keys), n_blocks=len(g),
                         partial_blocks=n_part,
                         net_j_per_query=ci["estimate"], lo=ci["lo"], hi=ci["hi"],
                         gross_j_per_query=g.gross_j_per_query.mean(), source=g.source.iloc[0],
                         is_mock=int(g.is_mock.max()) if "is_mock" in g else 0))
    sdf = pd.DataFrame(summ)
    for keys, g in edge.groupby(["device", "dataset", "model", "variant"]):
        device, ds, model, variant = keys
        cis = {}
        for p in s["precisions"]:
            vals = g[g.precision == p].net_j_per_query.dropna().tolist()
            cis[p] = (len(vals), S.bootstrap_ci(vals, n_boot=s["n_boot"], seed=s["seed"]))
        nmin = min(n for n, _ in cis.values())
        fam = f"H1b-energy/{ds}/{device}"
        n_part = int(((partial.device == device) & (partial.dataset == ds) & (partial.model == model)
                      & (partial.variant == variant)).sum()) if len(partial) else 0
        grp = _fmt_group(["model", "variant"], [model, variant])
        base = {"hypothesis": "H1b", "family": fam, "group": grp, "comparison": "F16 > Q8_0 > Q4_K_M (block energy/query)",
                "test": "block bootstrap CI", "n": nmin, "statistic": None, "p": None, "p_holm": None,
                "partial_blocks": n_part,
                "ci": "; ".join(f"{p}:[{c['lo']:.4g},{c['hi']:.4g}]" if c["lo"] is not None else f"{p}:NA"
                                for p, (_, c) in cis.items()), "effect": None, "holm": False}
        if nmin < 3:
            rows.append({**base, "decision": "not testable", "reason": f"not testable: {nmin} blocks (< 3) for a precision"})
            continue
        ordered_cis = [cis[p][1] for p in s["precisions"]]
        separated = all(lo["hi"] < hi["lo"] for hi, lo in zip(ordered_cis, ordered_cis[1:]))
        rows.append({**base, "decision": "supported" if separated else "not supported", "reason": None})
    return rows, sdf


def h2a(df: pd.DataFrame, s: dict) -> list[dict]:
    rows = []
    acc = df[(df.plan == s["acc_plan"]) & (df.strategy.isin(s["ft"]))]
    for keys, g in acc.groupby(["dataset", "device", "model", "precision"]):
        ds, device, model, prec = keys
        wide, removed = _pivot(g, "strategy", s["ft"], "correct")
        accs = [float(wide[c].mean()) if c in wide else None for c in s["ft"]]
        for cand in ("lora", "qlora"):
            grp = _fmt_group(["model", "precision"], [model, prec])
            fam = f"H2a/{ds}/{device}"
            if cand not in wide or "fft" not in wide:
                rows.append({"hypothesis": "H2a", "family": fam, "group": grp, "comparison": f"{cand} vs fft",
                             "test": "newcombe-10 non-inferiority", "decision": "not testable", "p": None,
                             "p_holm": None, "reason": "missing configs", "holm": False})
                continue
            mc = S.mcnemar_exact(wide[cand].tolist(), wide["fft"].tolist())
            ni = S.noninferiority(wide[cand].tolist(), wide["fft"].tolist(), s["delta"], s["alpha"])
            # H2a exception (DESIGN.md 2): few discordant pairs is agreement, so only the accuracy floor applies.
            reason = S.floor_check(None, None, accs, s["min_disc"], s["min_acc"])
            if reason:
                decision = "not testable"
            else:
                decision = "supported (non-inferior)" if ni["noninferior"] else "not supported"
            rows.append({"hypothesis": "H2a", "family": fam, "group": grp, "comparison": f"{cand} - fft >= -{s['delta']:g}",
                         "test": "newcombe-10 non-inferiority (exact McNemar p reported)", "n": ni["n"],
                         "statistic": ni["diff"], "p": mc["p"], "p_holm": None,
                         "effect": f"b/c={mc['odds_ratio']}", "ci": f"[{ni['lo']:.4f}, {ni['hi']:.4f}]" if ni["lo"] is not None else None,
                         "decision": decision, "reason": reason, "removed": removed, "holm": False})
    return rows


def h2b(df: pd.DataFrame, s: dict, domains: Iterable[str]) -> list[dict]:
    rows = []
    ret = _expand_retention_base(df[(df.plan == s["ret_plan"]) & (df.dataset == "arc_easy")], domains)
    for keys, g in ret.groupby(["train_domain", "device", "model", "precision"]):
        dom, device, model, prec = keys
        if not dom:
            continue
        wide, removed = _pivot(g, "strategy", ["base", *s["ft"]], "correct")
        accs = [float(wide[c].mean()) for c in wide.columns]
        for cand in ("lora", "qlora"):
            grp = _fmt_group(["model", "precision"], [model, prec])
            fam = f"H2b/arc_easy-after-{dom}/{device}"
            if cand not in wide or "fft" not in wide:
                rows.append(_decision_row("H2b", fam, grp, f"{cand} > fft", {"test": "mcnemar-exact", "reason": "missing configs"}))
                continue
            res = S.mcnemar_exact(wide[cand].tolist(), wide["fft"].tolist(), alternative="greater")
            reason = S.floor_check(res["b"], res["c"], accs, s["min_disc"], s["min_acc"])
            if reason:
                res = {**res, "p": None, "reason": reason}
            rows.append(_decision_row("H2b", fam, grp, f"{cand} > fft (one-sided)", res,
                                      effect=f"diff={res.get('diff')}, b/c={res.get('odds_ratio')}", removed=removed,
                                      direction_ok=bool(res["b"] > res["c"])))
    return rows


def cochran_family(df: pd.DataFrame, s: dict, hyp: str, plan: str, factor: str, levels: list[Any],
                   group_keys: list[str], label: str, reference: str | None = None) -> tuple[list[dict], list[dict]]:
    rows, post = [], []
    sub = df[df.plan == plan]
    for keys, g in sub.groupby(["dataset", "device", *group_keys]):
        ds, device, *rest = keys
        wide, removed = _pivot(g, factor, levels, "correct")
        fam = f"{hyp}/{ds}/{device}"
        grp = _fmt_group(group_keys, rest)
        if wide.shape[1] < 3:
            rows.append(_decision_row(hyp, fam, grp, label, {"test": "cochran-q", "reason": "fewer than 3 configs"}))
            continue
        accs = [float(wide[c].mean()) for c in wide.columns]
        res = S.cochrans_q(wide.values)
        floor = S.floor_check(None, None, accs, s["min_disc"], s["min_acc"])
        if floor:
            res = {**res, "p": None, "reason": floor}
        rows.append(_decision_row(hyp, fam, grp, label, res, removed=removed,
                                  effect="; ".join(f"{c}={a:.3f}" for c, a in zip(wide.columns, accs))))
        labels = list(wide.columns)
        pairs = None
        if reference and reference in labels:
            r = labels.index(reference)
            pairs = [(i, r) for i in range(len(labels)) if i != r]
        for pr in S.posthoc_mcnemar(wide.values.tolist(), labels, pairs):
            pr["not_testable"] = S.floor_check(pr["b"], pr["c"], accs, s["min_disc"], s["min_acc"])
            post.append({"family": fam, "group": grp, "metric": "correct", **pr})
    return rows, post


def h3(df: pd.DataFrame, s: dict) -> list[dict]:
    rows = []
    ks = s["k_levels"]
    rag = df[df.strategy == "rag"]
    edge = rag[rag.tier == "edge"]
    acc = rag[rag.plan == s["acc_plan"]]
    for keys, g in edge.groupby(["device", "dataset", "model", "precision"]):
        device, ds, model, prec = keys
        grp = _fmt_group(["model", "precision"], [model, prec])
        fam = f"H3/{ds}/{device}"
        for metric in ("ttft_primary_s", "e2e_latency_s"):
            wide, removed = _pivot(g, "k", ks, metric)
            res = S.page_l(wide.values) if wide.shape[1] == len(ks) else {"test": "page-l", "reason": "missing K levels"}
            med = [float(wide[c].median()) for c in wide.columns] if wide.shape[1] == len(ks) else []
            rows.append(_decision_row("H3", fam, grp, f"{metric} increases with K ({'<'.join(map(str, ks))})", res,
                                      removed=removed, direction_ok=bool(S.ordered(med, "increasing"))))
        k_mid, k_hi = str(ks[-2]), str(ks[-1])
        wide, _ = _pivot(g, "k", [ks[-2], ks[-1]], "e2e_latency_s")
        wil = S.wilcoxon_paired(wide[k_hi], wide[k_mid], "greater") if {k_mid, k_hi} <= set(wide.columns) else \
            {"test": "wilcoxon", "reason": "missing K levels"}
        wil_row = _decision_row("H3", fam, grp, f"e2e K{k_hi} > K{k_mid} (one-sided)", wil,
                                effect=f"r_rb={wil.get('r_rb')}", direction_ok=bool((wil.get("r_rb") or 0) > 0))
        rows.append(wil_row)
        ga = acc[(acc.dataset == ds) & (acc.model == model) & (acc.precision == prec)]
        wa, _ = _pivot(ga, "k", ks, "correct")
        if wa.shape[1] < 3 or len(ks) != 3:
            rows.append({"hypothesis": "H3", "family": fam, "group": grp, "comparison": "diminishing accuracy returns",
                         "test": "McNemar K10 vs K5 + bootstrap", "decision": "not testable", "p": None,
                         "p_holm": None, "reason": "accuracy-of-record rag configs missing", "holm": False})
            continue
        k_lo = str(ks[0])
        mc = S.mcnemar_exact(wa[k_hi].tolist(), wa[k_mid].tolist())
        accs = [float(wa[c].mean()) for c in wa.columns]
        boot = S.bootstrap_diminishing(wa[k_lo].tolist(), wa[k_mid].tolist(), wa[k_hi].tolist(), s["n_boot"], s["seed"])
        floor = S.floor_check(mc["b"], mc["c"], accs, s["min_disc"], s["min_acc"])
        rows.append({"hypothesis": "H3", "family": fam, "group": grp, "comparison": "diminishing accuracy returns",
                     "test": "McNemar(K10 vs K5) + Wilcoxon e2e, or bootstrap CI", "n": mc["n"],
                     "statistic": boot["estimate"], "p": mc["p"], "p_holm": None,
                     "effect": f"acc K3/K5/K10={accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f}",
                     "ci": f"[{boot['lo']:.4f}, {boot['hi']:.4f}]" if boot["lo"] is not None else None,
                     "decision": None, "reason": floor, "_h3_wilcoxon": wil_row,
                     "_ci_below": boot["hi"] is not None and boot["hi"] < 0})
    return rows


def accuracy_parity(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    keys = ["model", "strategy", "precision", "dataset", "context_mode", "k"]
    ref = df[(df.plan == s["acc_plan"]) & df.error.isna()]
    out = []
    for k_vals, g in df[(df.tier == "edge") & df.error.isna()].groupby(keys + ["device", "config_id"]):
        sel = ref
        for k, v in zip(keys, k_vals[: len(keys)]):
            sel = sel[sel[k] == v]
        merged = g[["query_id", "parsed_answer"]].merge(sel[["query_id", "parsed_answer"]], on="query_id",
                                                         suffixes=("_edge", "_ref"))
        n = len(merged)
        agree = float((merged.parsed_answer_edge.fillna("<none>") == merged.parsed_answer_ref.fillna("<none>")).mean()) if n else None
        out.append({**dict(zip(keys + ["device", "config_id"], k_vals)), "n_common": n, "agreement": agree,
                    "flag_rerun_500": bool(agree is not None and agree < s["parity"]),
                    "reason": None if n else "no accuracy-of-record rows for this config"})
    return pd.DataFrame(out)


def adaptation_energy(training: pd.DataFrame, s: dict) -> pd.Series:
    """One-off adaptation energy per (model, strategy, dataset): every LR-grid selection run counts, because
    a practitioner pays for the whole grid to obtain the selected model."""
    if training.empty:
        return pd.Series(dtype=float)
    sel = training[training.seed == s["selection_seed"]]
    if sel.empty:
        sel = training
    return sel.groupby(["model", "strategy", "dataset"]).energy_j.sum(min_count=1)


def per_query_energy(cfg_summary: pd.DataFrame, block_summary: pd.DataFrame,
                     blocks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-config J/query: metered block-level net energy, else per-query net energy from a counter or power
    source. Configs whose only energy source is the Pi PMIC get no figure ("no metered energy")."""
    out = cfg_summary[["config_id", "plan", "model", "strategy", "variant", "precision", "device", "dataset", "k",
                       "accuracy", "net_energy_j_mean", "energy_j_mean"]].copy()
    out["j_per_query"] = out["net_energy_j_mean"]
    out["energy_basis"] = "per-query trapezoid/counter (net)"
    if not block_summary.empty:
        b = block_summary.groupby(["device", "dataset", "model", "variant", "precision"]).net_j_per_query.mean()
        for i, r in out.iterrows():
            key = (r.device, r.dataset, r.model, r.variant, r.precision)
            if key in b.index and not pd.isna(b[key]):
                out.at[i, "j_per_query"] = b[key]
                out.at[i, "energy_basis"] = "block-level (net)"
    if blocks is not None and not blocks.empty:
        unmetered = set(blocks.loc[~blocks.source.map(metered_source), "config_id"]) - set(
            blocks.loc[blocks.source.map(metered_source) & blocks.energy_j.notna(), "config_id"])
        mask = out.config_id.isin(unmetered)
        out.loc[mask, "j_per_query"] = None
        out.loc[mask, "energy_basis"] = "no metered energy"
    return out


def cost_tables(energy: pd.DataFrame, training: pd.DataFrame, tariff: dict, s: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t, g = tariff["tariff_zar_per_kwh"], tariff["grid_kgco2e_per_kwh"]
    cost = energy.copy()
    cost["zar_per_1000_queries"] = [S.cost_zar(e * 1000, t) if pd.notna(e) else None for e in cost.j_per_query]
    cost["kgco2e_per_1000_queries"] = [S.co2e_kg(e * 1000, g) if pd.notna(e) else None for e in cost.j_per_query]
    cost["zar_per_correct"] = [S.cost_zar(e, t) / a if pd.notna(e) and a else None
                               for e, a in zip(cost.j_per_query, cost.accuracy)]
    be = []
    adapt = adaptation_energy(training, s)
    for keys, grp in cost[cost.plan == s["edge_plan"]].groupby(["device", "dataset", "model", "precision"]):
        device, ds, model, prec = keys
        ragrows = grp[grp.strategy == "rag"]
        for _, ft in grp[grp.strategy.isin(s["ft"])].iterrows():
            a = adapt.get((model, ft.strategy, ds))
            for _, rg in ragrows.iterrows():
                res = S.break_even(None if a is None or pd.isna(a) else float(a), ft.j_per_query, rg.j_per_query)
                be.append({"device": device, "dataset": ds, "model": model, "precision": prec,
                           "fine_tuned": ft.strategy, "rag": rg.variant, "adaptation_j": a,
                           "ft_j_per_query": ft.j_per_query, "rag_j_per_query": rg.j_per_query,
                           "break_even_queries": res["queries"], "reason": res["reason"]})
    return cost, pd.DataFrame(be)


def recall_table(df: pd.DataFrame) -> pd.DataFrame:
    rag = df[(df.strategy == "rag") & df.retrieval_hit.notna()]
    if rag.empty:
        return pd.DataFrame()
    return rag.groupby(["dataset", "k"]).retrieval_hit.agg(["mean", "count"]).reset_index().rename(
        columns={"mean": "recall_at_k_all_gold", "count": "n"})


def analyze(conn: sqlite3.Connection, matrix: dict, tariff: dict, out_dir: str | Path | None = None,
            allow_mock: bool = False) -> dict[str, pd.DataFrame]:
    mock = db.has_mock(conn)
    if mock and not allow_mock:
        raise MockDataRefused("results DB contains is_mock rows; refusing to analyse (use --allow-mock for a "
                              "watermarked test analysis)")
    s = _settings(matrix)
    fr = load_frames(conn, matrix, allow_mock=allow_mock)
    df, blocks, training = fr["runs"], fr["blocks"], fr["training"]
    out: dict[str, pd.DataFrame] = {}
    decisions: list[dict] = []
    posthoc: list[dict] = []
    r, out["training_summary"] = h1a(training, s)
    decisions += r
    if not df.empty:
        out["config_summary"] = config_summary(df, s)
        r, p = h1b_memory(df, s)
        decisions += r
        posthoc += p
        r, out["energy_blocks_summary"] = block_energy(blocks, s)
        decisions += r
        decisions += h2a(df, s)
        domains = matrix["datasets"].get("arc_easy", {}).get("retention_after", [])
        decisions += h2b(df, s, domains)
        r, p = cochran_family(df, s, "SQ2", s["acc_plan"], "variant",
                              ["base", *s["ft"], f"rag-k{s['sq2_k']}"], ["model", "precision"],
                              "accuracy differs across base/fft/lora/qlora/rag", reference="base")
        decisions += r
        posthoc += p
        r, p = cochran_family(df, s, "SQ3", s["acc_plan"], "precision", s["precisions"], ["model", "variant"],
                              "accuracy differs across " + "/".join(s["precisions"]))
        decisions += r
        posthoc += p
        decisions += h3(df, s)
        out["accuracy_parity"] = accuracy_parity(df, s)
        out["recall_at_k"] = recall_table(df)
        energy = per_query_energy(out["config_summary"], out["energy_blocks_summary"], blocks)
        out["cost"], out["break_even"] = cost_tables(energy, training, tariff, s)
    unmeasured = (matrix.get("report") or {}).get("energy_not_measured")
    if unmeasured and not any(str(d.get("family", "")).startswith("H1b-energy") for d in decisions):
        decisions.append({"hypothesis": "H1b", "family": "H1b-energy", "group": "all", "comparison":
                          "F16 > Q8_0 > Q4_K_M (block energy/query)", "test": None, "n": 0, "statistic": None,
                          "p": None, "p_holm": None, "effect": None, "ci": None, "holm": False,
                          "decision": "not measured", "reason": str(unmeasured)})
    _apply_holm(decisions, s["alpha"])
    out["hypothesis_tests"] = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in decisions])
    out["posthoc"] = pd.DataFrame(posthoc)
    out["tariff"] = pd.DataFrame([tariff])
    if mock:
        for name, frame in out.items():
            if frame.empty:
                out[name] = pd.DataFrame({"watermark": [MOCK_WATERMARK]})
            else:
                frame.insert(0, "watermark", MOCK_WATERMARK)
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for name, frame in out.items():
            frame.to_csv(od / f"{name}.csv", index=False)
        log.info("analysis CSVs written to %s", od)
    return out
