import math

import numpy as np
import pytest
from scipy import stats as sps

from slmbench import stats as S


def test_holm():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([None, 0.02]) == [None, 0.02]
    assert S.holm([]) == []


def test_wilcoxon_and_rank_biserial():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [v - 0.5 - i * 0.1 for i, v in enumerate(x)]
    r = S.wilcoxon_paired(x, y)
    assert r["p"] == pytest.approx(sps.wilcoxon(x, y).pvalue) and r["r_rb"] == 1.0
    assert S.wilcoxon_paired(x, y, "greater")["p"] < r["p"]


@pytest.mark.parametrize("x,y,reason", [([1, 2, 3], [1, 2, 3], "all ties"), ([1], [2], "too few"),
                                        ([1, 2, 3], [1, 2, 4], "fewer than 2"), ([0, 0], [0, 0], "all ties")])
def test_wilcoxon_degenerate(x, y, reason):
    r = S.wilcoxon_paired(x, y)
    assert r["p"] is None and reason in r["reason"]


def test_friedman_kendall_and_posthoc():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, 20)
    m = np.column_stack([base + 2, base + 1, base])
    r = S.friedman(m)
    assert r["p"] < 0.001 and r["kendalls_w"] == pytest.approx(1.0)
    post = S.posthoc_wilcoxon(m, ["F16", "Q8_0", "Q4_K_M"])
    assert len(post) == 3 and all(p["p_holm"] >= p["p"] for p in post)


def test_friedman_degenerate():
    assert "ties" in S.friedman([[1, 1, 1], [2, 2, 2]])["reason"]
    assert "3 conditions" in S.friedman([[1, 2], [2, 1]])["reason"]
    assert "too few" in S.friedman([[1, 2, 3]])["reason"]
    assert S.friedman([[1, 2, float("nan")], [1, 2, 3]])["n"] == 1


def test_mcnemar_exact():
    a = [1] * 10 + [1] * 8 + [0] * 2
    b = [1] * 10 + [0] * 8 + [1] * 2
    r = S.mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (8, 2) and r["odds_ratio"] == 4.0
    assert r["p"] == pytest.approx(sps.binomtest(8, 10, 0.5).pvalue)
    g = S.mcnemar_exact(a, b, "greater")
    assert g["p"] == pytest.approx(sps.binomtest(8, 10, 0.5, alternative="greater").pvalue)
    assert S.mcnemar_exact([1, 0], [1, 0])["reason"] == "no discordant pairs"
    assert S.mcnemar_exact([1, 1], [0, 0])["odds_ratio"] == math.inf


def test_cochrans_q_matches_formula_and_degenerate():
    m = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0]]
    r = S.cochrans_q(m)
    assert r["statistic"] == pytest.approx(6.0) and r["df"] == 2
    assert r["p"] == pytest.approx(sps.chi2.sf(6.0, 2))
    assert S.cochrans_q([[1, 1, 1], [0, 0, 0]])["reason"]
    post = S.posthoc_mcnemar(m, ["a", "b", "c"], pairs=[(1, 0), (2, 0)])
    assert [(p["level_a"], p["level_b"]) for p in post] == [("b", "a"), ("c", "a")]


def test_newcombe_ci_properties():
    a = [1] * 12 + [1] * 9 + [0] * 2 + [0] * 1
    b = [1] * 12 + [0] * 9 + [1] * 2 + [0] * 1
    r = S.newcombe_paired_ci(a, b)
    assert r["diff"] == pytest.approx(7 / 24)
    assert r["lo"] < r["diff"] < r["hi"] and -1 <= r["lo"] and r["hi"] <= 1
    rev = S.newcombe_paired_ci(b, a)
    assert rev["lo"] == pytest.approx(-r["hi"]) and rev["hi"] == pytest.approx(-r["lo"])
    same = S.newcombe_paired_ci([1, 0, 1, 1], [1, 0, 1, 1])
    assert same["diff"] == 0 and same["lo"] < 0 < same["hi"]
    assert S.newcombe_paired_ci([], [])["reason"]


def test_noninferiority_decision():
    n = 400
    ref = [1] * 300 + [0] * 100
    cand = list(ref)
    assert S.noninferiority(cand, ref, 0.05)["noninferior"] is True
    worse = [0 if i < 60 else v for i, v in enumerate(ref)]
    assert S.noninferiority(worse, ref, 0.05)["noninferior"] is False
    assert len(ref) == n


def test_page_l():
    m = [[1, 2, 3], [1.5, 2.5, 3.5], [1, 3, 4], [2, 3, 5], [1, 2, 2.5]]
    r = S.page_l(m)
    assert r["p"] == pytest.approx(sps.page_trend_test(np.array(m)).pvalue) and r["p"] < 0.05
    assert S.page_l([[1, 1, 1], [2, 2, 2]])["reason"]
    assert S.page_l([[1, 2]])["reason"]


def test_bootstrap_ci_and_degenerate():
    r = S.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=2000, seed=42)
    assert r["lo"] <= r["estimate"] == 0.75 <= r["hi"]
    assert S.bootstrap_ci([1, 0, 1], n_boot=500, seed=1) == S.bootstrap_ci([1, 0, 1], n_boot=500, seed=1)
    assert S.bootstrap_ci([])["reason"] == "no observations"
    z = S.bootstrap_ci([0, 0, 0, 0])
    assert z["estimate"] == z["lo"] == z["hi"] == 0.0


def test_bootstrap_diminishing():
    c3 = [0] * 50 + [1] * 50
    c5 = [0] * 20 + [1] * 80
    c10 = [0] * 18 + [1] * 82
    r = S.bootstrap_diminishing(c3, c5, c10, n_boot=2000)
    assert r["estimate"] == pytest.approx((0.82 - 0.8) / 5 - (0.8 - 0.5) / 2)
    assert r["hi"] < 0


def test_floor_rule():
    assert "discordant" in S.floor_check(4, 5, [0.5, 0.6])
    assert "below 5%" in S.floor_check(20, 20, [0.01, 0.04])
    assert S.floor_check(8, 4, [0.4, 0.5]) is None
    assert S.floor_check(None, None, []) is None


def test_describe_cv_ordered():
    d = S.describe([1, 2, 3, 4])
    assert d["median"] == 2.5 and d["iqr"] == pytest.approx(1.5) and d["n"] == 4
    assert S.describe([])["mean"] is None
    assert S.coefficient_of_variation([10, 10, 10]) == 0
    assert S.coefficient_of_variation([5]) is None
    assert S.ordered([3, 2, 1], "decreasing") and not S.ordered([1, 1, 2], "increasing")
    assert S.ordered([1, None], "increasing") is None


def test_break_even_cost_co2e():
    r = S.break_even(3.6e6, 1.0, 3.0)
    assert r["queries"] == pytest.approx(1.8e6)
    u = S.break_even(1000.0, 3.0, 2.0)
    assert u["queries"] is None and u["reason"].startswith("undefined")
    assert S.break_even(1000.0, 2.0, 2.0)["reason"].startswith("undefined")
    assert S.break_even(None, 1, 2)["reason"] == "missing inputs"
    assert S.cost_zar(3.6e6, 4.14) == pytest.approx(4.14)
    assert S.co2e_kg(7.2e6, 0.96) == pytest.approx(1.92)
