"""Inline USB-C power meter: cumulative mWh readings entered by hand or imported from the meter's CSV export."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

TIME_KEYS = ("timestamp", "time", "datetime", "t")
MWH_KEYS = ("mwh", "energy_mwh", "cumulative_mwh")
WH_KEYS = ("wh", "energy_wh")
POWER_KEYS = ("power_w", "w", "watts", "power")


class NaiveTimestampError(ValueError):
    pass


def parse_tz(tz: str | None) -> tzinfo | None:
    """'+02:00' / '-0530' style offsets or an IANA name such as 'Africa/Johannesburg'."""
    if not tz:
        return None
    t = tz.strip()
    if t.upper() in ("UTC", "Z"):
        return timezone.utc
    if t[0] in "+-":
        sign = 1 if t[0] == "+" else -1
        hh, mm = (t[1:].split(":") + ["0"])[:2] if ":" in t else (t[1:3], t[3:5] or "0")
        return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
    return ZoneInfo(t)


def _to_epoch(v: str, tz: tzinfo | None = None) -> float:
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            if tz is None:
                raise NaiveTimestampError(f"timestamp {v!r} has no time zone; pass --tz (e.g. +02:00)")
            dt = dt.replace(tzinfo=tz)
        return dt.timestamp()


def _pick(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    low = {k.strip().lower(): v for k, v in row.items() if k}
    for k in keys:
        if k in low and str(low[k]).strip():
            return low[k]
    return None


def read_meter_csv(path: str | Path, tz: str | None = None) -> list[dict[str, float]]:
    """Rows of {t, mwh?, power_w?}, sorted by time. Wh columns are converted to mWh.

    Timestamps without an offset are rejected unless ``tz`` names the meter clock's zone.
    """
    zone = parse_tz(tz)
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = _pick(row, TIME_KEYS)
            if t is None:
                continue
            rec: dict[str, float] = {"t": _to_epoch(t, zone)}
            mwh, wh, p = _pick(row, MWH_KEYS), _pick(row, WH_KEYS), _pick(row, POWER_KEYS)
            if mwh is not None:
                rec["mwh"] = float(mwh)
            elif wh is not None:
                rec["mwh"] = float(wh) * 1000.0
            if p is not None:
                rec["power_w"] = float(p)
            out.append(rec)
    out.sort(key=lambda r: r["t"])
    if not out:
        log.warning("%s: no usable rows", path)
    return out


def cumulative_at(rows: list[dict[str, float]], t: float) -> float | None:
    """Cumulative mWh at time t by linear interpolation; None outside the recorded span."""
    pts = [(r["t"], r["mwh"]) for r in rows if "mwh" in r]
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    for (ta, ea), (tb, eb) in zip(pts, pts[1:]):
        if ta <= t <= tb:
            return ea if tb == ta else ea + (eb - ea) * (t - ta) / (tb - ta)
    return pts[-1][1]


def iso_to_epoch(s: str) -> float:
    return _to_epoch(s)


def block_readings(rows: list[dict[str, float]], started_at: str, ended_at: str,
                   idle_started_at: str | None = None) -> dict[str, Any]:
    out = {"meter_start_mwh": cumulative_at(rows, iso_to_epoch(started_at)),
           "meter_end_mwh": cumulative_at(rows, iso_to_epoch(ended_at))}
    if idle_started_at:
        out["idle_meter_start_mwh"] = cumulative_at(rows, iso_to_epoch(idle_started_at))
    return out
