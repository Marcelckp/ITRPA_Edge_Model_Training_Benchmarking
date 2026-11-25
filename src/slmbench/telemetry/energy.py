"""Energy arithmetic: trapezoid integration over a window, idle-baseline subtraction, unit conversion."""

from __future__ import annotations

from typing import Sequence

J_PER_KWH = 3.6e6
J_PER_MWH = 3.6  # milliwatt-hour


def _interp(t: float, t0: float, p0: float, t1: float, p1: float) -> float:
    if t1 == t0:
        return p0
    return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


def integrate_trapezoid(times: Sequence[float], powers: Sequence[float], start: float, end: float) -> float | None:
    """Energy (J) over [start, end] from (t, W) samples, linearly interpolated at the window edges.

    Outside the sampled range the nearest sample is held constant. Returns None without samples.
    """
    pts = sorted((float(t), float(p)) for t, p in zip(times, powers) if p is not None)
    if not pts or end <= start:
        return None if not pts else 0.0
    if len(pts) == 1:
        return pts[0][1] * (end - start)

    def power_at(t: float) -> float:
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return _interp(t, ta, pa, tb, pb)
        return pts[-1][1]

    knots = [start] + [t for t, _ in pts if start < t < end] + [end]
    values = [power_at(t) for t in knots]
    return sum((knots[i + 1] - knots[i]) * (values[i] + values[i + 1]) / 2 for i in range(len(knots) - 1))


def mean_power(times: Sequence[float], powers: Sequence[float], start: float, end: float) -> float | None:
    e = integrate_trapezoid(times, powers, start, end)
    if e is None or end <= start:
        return None
    return e / (end - start)


def net_energy(gross_j: float | None, idle_power_w: float | None, duration_s: float | None) -> float | None:
    """Gross minus idle power x duration. Not clipped at zero, so noise is not biased upward."""
    if gross_j is None or idle_power_w is None or duration_s is None:
        return None
    return gross_j - idle_power_w * duration_s


def kwh(energy_j: float) -> float:
    return energy_j / J_PER_KWH
