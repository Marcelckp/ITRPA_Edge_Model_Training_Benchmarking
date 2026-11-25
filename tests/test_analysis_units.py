"""Decision-rule units on hand-built frames (no runner)."""

import numpy as np
import pandas as pd

from slmbench import analysis as A
from slmbench.advise import Constraints, rank


def _s():
    return A._settings({"analysis": {"bootstrap_resamples": 500}})


def _acc_frame(fft, lora, qlora, model="m", ds="pubmedqa"):
    rows = []
    for strat, vals in (("fft", fft), ("lora", lora), ("qlora", qlora)):
        for i, v in enumerate(vals):
            rows.append({"plan": "accuracy", "strategy": strat, "dataset": ds, "device": "gpu", "model": model,
                         "precision": "Q4_K_M", "query_id": f"q{i}", "correct": v, "error": None})
    return pd.DataFrame(rows)


def test_h2a_noninferior_supported():
    rng = np.random.default_rng(1)
    base = (rng.random(400) < 0.6).astype(int)
    flip = base.copy()
    idx = rng.choice(400, 30, replace=False)
    flip[idx] = 1 - flip[idx]
    rows = A.h2a(_acc_frame(base, flip, base), _s())
    lora = next(r for r in rows if r["comparison"].startswith("lora"))
    q = next(r for r in rows if r["comparison"].startswith("qlora"))
    assert lora["decision"] in ("supported (non-inferior)", "not supported") and lora["ci"]
    # H2a exception: 0 discordant pairs is agreement, so identical-to-FFT is non-inferior, not "not testable".
    assert q["decision"] == "supported (non-inferior)" and q["reason"] is None


def test_h2a_floor_all_below_5_percent():
    z = [0] * 100
    rows = A.h2a(_acc_frame(z, z, [1] * 3 + [0] * 97), _s())
    assert all(r["decision"] == "not testable" and "below" in r["reason"] for r in rows)


def test_listwise_removal_counts_errors():
    df = _acc_frame([1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1])
    df.loc[(df.strategy == "lora") & (df.query_id == "q2"), "error"] = "boom"
    wide, removed = A._pivot(df, "strategy", ["fft", "lora", "qlora"], "correct")
    assert removed == 1 and "q2" not in wide.index and len(wide) == 3


def test_holm_family_and_decisions():
    rows = [{"hypothesis": "H", "family": "F1", "p": 0.01, "decision": None, "reason": None},
            {"hypothesis": "H", "family": "F1", "p": 0.04, "decision": None, "reason": None},
            {"hypothesis": "H", "family": "F2", "p": 0.04, "decision": None, "reason": None},
            {"hypothesis": "H", "family": "F2", "p": None, "decision": None, "reason": "not testable: x"}]
    A._apply_holm(rows, 0.05)
    assert [r["p_holm"] for r in rows] == [0.02, 0.04, 0.04, None]
    assert [r["decision"] for r in rows] == ["reject H0", "reject H0", "reject H0", "not testable"]


def test_advisor_ranking_and_constraints():
    cands = pd.DataFrame([
        {"model": "a", "variant": "lora", "precision": "Q4_K_M", "dataset": "d", "accuracy": 0.7, "p95_e2e_s": 2.0,
         "peak_mem_mb": 900.0, "monthly_zar": 5.0},
        {"model": "b", "variant": "rag-k3", "precision": "Q8_0", "dataset": "d", "accuracy": 0.8, "p95_e2e_s": 3.0,
         "peak_mem_mb": 1500.0, "monthly_zar": 4.0},
        {"model": "c", "variant": "base", "precision": "F16", "dataset": "d", "accuracy": 0.9, "p95_e2e_s": 9.0,
         "peak_mem_mb": 800.0, "monthly_zar": 1.0},
        {"model": "d", "variant": "fft", "precision": "F16", "dataset": "d", "accuracy": 0.4, "p95_e2e_s": 1.0,
         "peak_mem_mb": 700.0, "monthly_zar": 0.5},
    ])
    c = Constraints(device="x", ram_mb=1000, latency_budget_s=5.0, min_accuracy=0.5)
    feas, infeas = rank(cands, c)
    assert list(feas.model) == ["a"] and feas["rank"].tolist() == [1]
    reasons = dict(zip(infeas.model, infeas.infeasible_reason))
    assert "memory" in reasons["b"] and "latency" in reasons["c"] and "accuracy" in reasons["d"]
    feas2, _ = rank(cands, Constraints(device="x", objective="accuracy"))
    assert list(feas2.model) == ["c", "b", "a", "d"]
    feas3, _ = rank(cands, Constraints(device="x"))
    assert list(feas3.model) == ["d", "c", "b", "a"]
    flagged = cands.assign(parity_flag=[False, True, False, False])
    feas4, infeas4 = rank(flagged, Constraints(device="x"))
    assert "b" not in list(feas4.model) and "re-run on 500" in infeas4.infeasible_reason.iloc[0]
