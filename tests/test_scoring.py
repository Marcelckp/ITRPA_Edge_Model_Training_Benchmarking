import pytest

from slmbench import scoring as sc


@pytest.mark.parametrize("text,expected", [
    ("Yes.", "yes"), ("maybe", "maybe"), ("  NO, because", "no"), ("The answer is maybe.", "maybe"),
    ("garbage output", None), ("", None), (None, None), ("yesterday", None),
])
def test_pubmedqa_parse(text, expected):
    assert sc.parse_pubmedqa(text) == expected


def test_pubmedqa_accuracy_and_macro_f1():
    golds = ["yes", "no", "maybe", "yes"]
    preds = ["yes", "no", None, "no"]
    assert sc.accuracy([g == p for g, p in zip(golds, preds)]) == 0.5
    f1 = sc.macro_f1(golds, preds)
    assert f1 == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    rec = sc.per_class_recall(golds, preds)
    assert rec == {"yes": 0.5, "no": 1.0, "maybe": 0.0}
    assert sc.macro_f1([], []) is None


@pytest.mark.parametrize("text,value,pct", [
    ("12.5%", 12.5, True), ("$1,234", 1234.0, False), ("-3.0", -3.0, False), ("(4.2)", -4.2, False),
    ("\u22127", -7.0, False), ("The answer is 0.5 percent", 0.5, True),
])
def test_parse_number(text, value, pct):
    n = sc.parse_number(text)
    assert n.value == pytest.approx(value) and n.is_percent == pct


def test_finqa_numeric_rules():
    assert sc.finqa_correct("12.5%", 0.125).correct
    assert sc.finqa_correct("12.5%", "12.5%").correct
    assert sc.finqa_correct("$1,234", 1234.0).correct
    assert sc.finqa_correct("-3.0", "-3").correct
    assert sc.finqa_correct("12.46", "12.5").correct  # rounds to the gold's 1 decimal
    assert not sc.finqa_correct("12.3", "12.5").correct
    assert sc.finqa_correct("1000", 1005.0).correct  # within 1e-2 relative
    assert not sc.finqa_correct("1000", 1020.0).correct
    assert sc.finqa_correct("0", 0.0).correct
    s = sc.finqa_correct("no number here", 5.0)
    assert not s.parse_ok and not s.correct


def test_finqa_yes_no_golds_exact_match_only():
    assert sc.finqa_correct("yes", "yes").correct
    assert sc.finqa_correct("Yes.", "yes").correct
    assert not sc.finqa_correct("yes, it increased by 5", "yes").correct
    assert not sc.finqa_correct("no", "yes").correct
    s = sc.finqa_correct("1", "no")
    assert not s.correct and not s.parse_ok


@pytest.mark.parametrize("text,expected", [
    ("B", "B"), ("(C)", "C"), ("Answer: D", "D"), ("The answer is A.", "A"), ("A. toy option", "A"),
    ("I pick (B) because", "B"), ("none", None),
])
def test_arc_parse(text, expected):
    assert sc.parse_arc(text, ["A", "B", "C", "D"]) == expected


def test_score_dispatch():
    assert sc.score("arc_easy", "C", {"gold": "C", "choices": [{"label": x} for x in "ABCD"]}).correct
    with pytest.raises(ValueError):
        sc.score("nope", "x", {})
