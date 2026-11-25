"""Statistical tests and effect sizes (DESIGN.md 2 and 3.7).

Every function returns a dict. Degenerate input (all ties, all zero, too few pairs) yields
``reason`` set and ``p`` None instead of an exception.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sps


def _result(test: str, **kw: Any) -> dict[str, Any]:
    out = {"test": test, "statistic": None, "p": None, "n": None, "reason": None}
    out.update(kw)
    return out


def holm(pvals: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values; None entries are skipped and stay None."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    m = len(idx)
    adj: list[float | None] = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        val = min(1.0, (m - rank) * float(pvals[i]))
        running = max(running, val)
        adj[i] = running
    return adj


def _paired(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over non-zero differences."""
    a, b = _paired(x, y)
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return None
    ranks = sps.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def wilcoxon_paired(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided") -> dict[str, Any]:
    """Wilcoxon signed-rank on x - y; alternative 'greater' tests x > y."""
    a, b = _paired(x, y)
    d = a - b
    nz = int(np.count_nonzero(d))
    res = _result("wilcoxon", n=int(a.size), n_nonzero=nz, alternative=alternative, r_rb=None)
    if a.size < 2:
        res["reason"] = "too few pairs"
        return res
    if nz == 0:
        res["reason"] = "all differences are zero (all ties)"
        return res
    if nz < 2:
        res["reason"] = "fewer than 2 non-zero differences"
        return res
    w = sps.wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
    res.update(statistic=float(w.statistic), p=float(w.pvalue), r_rb=rank_biserial(a, b))
    return res


def _complete(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        return np.empty((0, 0))
    return m[~np.isnan(m).any(axis=1)]


def friedman(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Friedman test across k columns (conditions) with n complete rows (blocks); Kendall's W effect size."""
    m = _complete(matrix)
    n, k = (m.shape if m.size else (0, 0))
    res = _result("friedman", n=int(n), k=int(k), kendalls_w=None)
    if k < 3:
        res["reason"] = "Friedman needs at least 3 conditions"
        return res
    if n < 2:
        res["reason"] = "too few complete blocks"
        return res
    if np.all(m == m[:, :1]):
        res["reason"] = "all ties within every block"
        return res
    stat, p = sps.friedmanchisquare(*[m[:, j] for j in range(k)])
    if not np.isfinite(stat):
        res["reason"] = "statistic undefined (ties)"
        return res
    res.update(statistic=float(stat), p=float(p), kendalls_w=float(stat / (n * (k - 1))))
    return res


def kendalls_w(matrix: Sequence[Sequence[float]]) -> float | None:
    return friedman(matrix)["kendalls_w"]


def posthoc_wilcoxon(matrix: Sequence[Sequence[float]], labels: Sequence[str],
                     alternative: str = "two-sided") -> list[dict[str, Any]]:
    m = _complete(matrix)
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = wilcoxon_paired(m[:, i], m[:, j], alternative) if m.size else _result("wilcoxon", reason="no data")
            out.append({**r, "level_a": labels[i], "level_b": labels[j]})
    for r, p in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = p
    return out


def _binary(x: Sequence[Any]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(bool(v)) for v in x], dtype=float)


def discordant(a: Sequence[Any], b: Sequence[Any]) -> tuple[int, int, int, int]:
    """(both correct, a only, b only, neither) over pairs where both are observed."""
    x, y = _paired(_binary(a), _binary(b))
    return (int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()),
            int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum()))


def mcnemar_exact(a: Sequence[Any], b: Sequence[Any], alternative: str = "two-sided") -> dict[str, Any]:
    """Exact (binomial) McNemar on discordant pairs. alternative 'greater' tests P(a correct) > P(b correct)."""
    n11, n10, n01, n00 = discordant(a, b)
    n = n11 + n10 + n01 + n00
    odds = (n10 / n01) if n01 else (math.inf if n10 else None)
    res = _result("mcnemar-exact", n=n, b=n10, c=n01, odds_ratio=odds, alternative=alternative,
                  diff=(n10 - n01) / n if n else None)
    if n10 + n01 == 0:
        res["reason"] = "no discordant pairs"
        return res
    bt = sps.binomtest(n10, n10 + n01, 0.5, alternative=alternative)
    res.update(statistic=float(n10), p=float(bt.pvalue))
    return res


def cochrans_q(matrix: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Cochran's Q for k related binary samples (rows = items, columns = configs)."""
    m = _complete([[np.nan if v is None else float(bool(v)) for v in row] for row in matrix])
    n, k = (m.shape if m.size else (0, 0))
    res = _result("cochran-q", n=int(n), k=int(k), df=max(int(k) - 1, 0))
    if k < 3:
        res["reason"] = "Cochran's Q needs at least 3 conditions"
        return res
    if n < 2:
        res["reason"] = "too few complete items"
        return res
    col = m.sum(axis=0)
    row = m.sum(axis=1)
    total = m.sum()
    denom = k * total - (row ** 2).sum()
    if denom == 0:
        res["reason"] = "every item has identical outcomes across configs"
        return res
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / denom
    res.update(statistic=float(q), p=float(sps.chi2.sf(q, k - 1)))
    return res


def posthoc_mcnemar(matrix: Sequence[Sequence[Any]], labels: Sequence[str],
                    pairs: Sequence[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    m = np.asarray(matrix, dtype=object)
    pairs = pairs or [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    out = []
    for i, j in pairs:
        r = mcnemar_exact(list(m[:, i]), list(m[:, j])) if m.size else _result("mcnemar-exact", reason="no data")
        out.append({**r, "level_a": labels[i], "level_b": labels[j]})
    for r, p in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = p
    return out


def wilson(x: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired_ci(a: Sequence[Any], b: Sequence[Any], alpha: float = 0.05) -> dict[str, Any]:
    """CI for P(a correct) - P(b correct) on paired binary data: Newcombe (1998) method 10."""
    n11, n10, n01, n00 = discordant(a, b)
    n = n11 + n10 + n01 + n00
    res = {"method": "newcombe-10", "n": n, "diff": None, "lo": None, "hi": None, "reason": None}
    if n == 0:
        res["reason"] = "no paired observations"
        return res
    z = float(sps.norm.ppf(1 - alpha / 2))
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n, z)
    l2, u2 = wilson(n11 + n01, n, z)
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    # Newcombe (1998) continuity-corrected phi; without it b = c = 0 at p = 0.5 gives a zero-width interval.
    ad, bc = n11 * n00, n10 * n01
    num = max(ad - bc - n / 2, 0.0) if ad > bc else ad - bc
    phi = num / math.sqrt(den) if den > 0 else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = d + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    res.update(diff=d, lo=max(-1.0, lo), hi=min(1.0, hi))
    return res


def noninferiority(candidate: Sequence[Any], reference: Sequence[Any], delta: float = 0.05,
                   alpha: float = 0.05) -> dict[str, Any]:
    """Candidate is non-inferior if the lower Newcombe bound of (candidate - reference) exceeds -delta."""
    ci = newcombe_paired_ci(candidate, reference, alpha)
    ci["delta"] = delta
    ci["noninferior"] = None if ci["lo"] is None else bool(ci["lo"] > -delta)
    return ci


def page_l(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Page's L trend test; columns must be ordered by the predicted increase."""
    m = _complete(matrix)
    n, k = (m.shape if m.size else (0, 0))
    res = _result("page-l", n=int(n), k=int(k))
    if k < 3:
        res["reason"] = "Page's L needs at least 3 ordered conditions"
        return res
    if n < 2:
        res["reason"] = "too few complete blocks"
        return res
    if np.all(m == m[:, :1]):
        res["reason"] = "all ties within every block"
        return res
    r = sps.page_trend_test(m)
    res.update(statistic=float(r.statistic), p=float(r.pvalue), method=r.method)
    return res


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 10000,
                 seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    res = {"estimate": None, "lo": None, "hi": None, "n": int(v.size), "reason": None}
    if v.size == 0:
        res["reason"] = "no observations"
        return res
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(axis=1)
    res.update(estimate=float(stat(v)), lo=float(np.quantile(boots, alpha / 2)),
               hi=float(np.quantile(boots, 1 - alpha / 2)))
    return res


def bootstrap_diminishing(c3: Sequence[Any], c5: Sequence[Any], c10: Sequence[Any], n_boot: int = 10000,
                          seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    """CI of (Acc10 - Acc5)/5 - (Acc5 - Acc3)/2, resampling items jointly across the three K levels."""
    m = _complete([[np.nan if v is None else float(bool(v)) for v in row] for row in zip(c3, c5, c10)])
    res = {"estimate": None, "lo": None, "hi": None, "n": int(m.shape[0]) if m.size else 0, "reason": None}
    if not m.size:
        res["reason"] = "no complete items"
        return res

    def f(a: np.ndarray) -> np.ndarray:
        acc = a.mean(axis=-2)
        return (acc[..., 2] - acc[..., 1]) / 5 - (acc[..., 1] - acc[..., 0]) / 2

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[0], size=(n_boot, m.shape[0]))
    boots = f(m[idx])
    res.update(estimate=float(f(m)), lo=float(np.quantile(boots, alpha / 2)), hi=float(np.quantile(boots, 1 - alpha / 2)))
    return res


def floor_check(b: int | None, c: int | None, accuracies: Sequence[float | None], min_discordant: int = 10,
                min_accuracy: float = 0.05) -> str | None:
    """DESIGN.md floor-effect rule: the reason a comparison is 'not testable', or None if testable."""
    accs = [a for a in accuracies if a is not None]
    if accs and all(a < min_accuracy for a in accs):
        return f"not testable: every config below {min_accuracy:.0%} accuracy"
    if b is not None and c is not None and b + c < min_discordant:
        return f"not testable: {b + c} discordant pairs (< {min_discordant})"
    return None


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None, "iqr": None,
                "min": None, "max": None}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(v.min()), "max": float(v.max())}


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    d = describe(values)
    if d["n"] is None or d["n"] < 2 or not d["mean"]:
        return None
    return float(d["sd"] / d["mean"])


def cost_zar(energy_j: float | None, tariff_zar_per_kwh: float) -> float | None:
    return None if energy_j is None else energy_j / 3.6e6 * tariff_zar_per_kwh


def co2e_kg(energy_j: float | None, grid_kg_per_kwh: float) -> float | None:
    return None if energy_j is None else energy_j / 3.6e6 * grid_kg_per_kwh


def break_even(adapt_j_ft: float | None, per_query_j_ft: float | None, per_query_j_rag: float | None,
               adapt_j_rag: float = 0.0) -> dict[str, Any]:
    """Query volume above which fine-tuning's lower per-query energy repays its one-off adaptation energy."""
    res = {"queries": None, "reason": None}
    if None in (adapt_j_ft, per_query_j_ft, per_query_j_rag):
        res["reason"] = "missing inputs"
        return res
    saving = per_query_j_rag - per_query_j_ft
    if saving <= 0:
        res["reason"] = "undefined: fine-tuned per-query cost >= RAG per-query cost"
        return res
    res["queries"] = max(0.0, (adapt_j_ft - adapt_j_rag) / saving)
    return res


def ordered(values: Sequence[float | None], direction: str) -> bool | None:
    """Whether values are strictly ordered (``decreasing`` or ``increasing``) in the given sequence."""
    if any(v is None for v in values) or len(values) < 2:
        return None
    pairs = list(zip(values, values[1:]))
    return all(a > b for a, b in pairs) if direction == "decreasing" else all(a < b for a, b in pairs)
