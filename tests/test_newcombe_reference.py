"""Reference values from Fagerland's R package contingencytables, Newcombe_square_and_add_CI_paired_2x2."""

import pytest

from slmbench import stats as S


def _pairs(n11, n12, n21, n22):
    a = [1] * n11 + [1] * n12 + [0] * n21 + [0] * n22
    b = [1] * n11 + [0] * n12 + [1] * n21 + [0] * n22
    return a, b


@pytest.mark.parametrize("table,expected", [
    ((1, 1, 7, 12), (-0.2857, -0.5069, -0.0256)),   # Bentur
    ((59, 6, 16, 80), (-0.0621, -0.1186, -0.0046)),  # Cavo
    ((1, 0, 0, 1), (0.0, -0.5734, 0.5734)),          # b = c = 0
])
def test_newcombe_matches_contingencytables(table, expected):
    r = S.newcombe_paired_ci(*_pairs(*table))
    assert (r["diff"], r["lo"], r["hi"]) == pytest.approx(expected, abs=1e-4)
