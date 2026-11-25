import time

import pytest

from slmbench.telemetry import MockPowerSource, TelemetrySession
from slmbench.telemetry.energy import integrate_trapezoid, mean_power, net_energy
from slmbench.telemetry.pi_pmic import is_throttled_now, parse_pmic, parse_throttled
from slmbench.telemetry.rapl import RaplSource
from slmbench.telemetry.tegrastats import parse_tegrastats
from slmbench.telemetry.usb_meter_csv import block_readings, cumulative_at, read_meter_csv


def test_trapezoid_constant_and_ramp():
    assert integrate_trapezoid([0, 1, 2], [5, 5, 5], 0, 2) == pytest.approx(10.0)
    assert integrate_trapezoid([0, 2], [0, 4], 0, 2) == pytest.approx(4.0)
    # window edges interpolated: power at 0.5 is 1, at 1.5 is 3
    assert integrate_trapezoid([0, 2], [0, 4], 0.5, 1.5) == pytest.approx(2.0)
    assert integrate_trapezoid([5], [3], 0, 2) == pytest.approx(6.0)
    assert integrate_trapezoid([], [], 0, 1) is None
    assert mean_power([0, 2], [2, 2], 0, 2) == pytest.approx(2.0)


def test_idle_subtraction():
    assert net_energy(10.0, 2.0, 3.0) == pytest.approx(4.0)
    assert net_energy(1.0, 2.0, 3.0) == pytest.approx(-5.0)  # not clipped
    assert net_energy(None, 2.0, 1.0) is None


def test_pmic_parse_partial_rails():
    out = """ VDD_CORE_A current(7)=2.00000000A
 VDD_CORE_V volt(15)=0.80000000V
 DDR_VDD2_A current(4)=0.50000000A
 DDR_VDD2_V volt(12)=1.10000000V
 EXT5V_V volt(24)=5.10000000V"""
    assert parse_pmic(out) == pytest.approx(2.0 * 0.8 + 0.5 * 1.1)
    assert parse_pmic("nothing") is None


def test_throttle_bits():
    assert parse_throttled("throttled=0x50005") == 0x50005
    assert is_throttled_now(0x50000) is False
    assert is_throttled_now(0x4) is True
    assert is_throttled_now(None) is None


def test_tegrastats_parse():
    w, c = parse_tegrastats("RAM 2000/7000MB CPU@45.5C VDD_IN 5123mW/5000mW VDD_CPU_GPU_CV 900mW/900mW")
    assert w == pytest.approx(5.123) and c == pytest.approx(45.5)


def test_rapl_counter_wraps(tmp_path):
    zone = tmp_path / "intel-rapl:0"
    zone.mkdir()
    (zone / "max_energy_range_uj").write_text("1000000")
    (zone / "energy_uj").write_text("900000")
    src = RaplSource("intel-rapl:0", root=tmp_path)
    assert src.available() and src.has_counter and src.name == "rapl-package"
    e0 = src.energy_j()
    (zone / "energy_uj").write_text("100000")  # wrapped
    assert src.energy_j() - e0 == pytest.approx(0.200001)


def test_rapl_missing_degrades(tmp_path):
    assert not RaplSource("intel-rapl:0", root=tmp_path).available()


def test_usb_meter_csv(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("timestamp,Wh\n2026-01-01T00:00:00Z,1.000\n2026-01-01T00:01:40Z,1.100\n")
    rows = read_meter_csv(p)
    assert rows[0]["mwh"] == pytest.approx(1000.0)
    t0 = rows[0]["t"]
    assert cumulative_at(rows, t0 + 50) == pytest.approx(1050.0)
    assert cumulative_at(rows, t0 - 1) is None
    rd = block_readings(rows, "2026-01-01T00:00:10+00:00", "2026-01-01T00:01:30+00:00")
    assert rd["meter_end_mwh"] - rd["meter_start_mwh"] == pytest.approx(80.0)


def test_session_window_and_idle():
    s = TelemetrySession(power=MockPowerSource(4.0), temperature=MockPowerSource(), interval_s=0.01)
    s.start()
    try:
        time.sleep(0.05)
        mark = s.begin()
        time.sleep(0.1)
        st = s.end(mark)
        idle_w, src = s.measure_idle(0.05)
    finally:
        s.stop()
    assert st.energy_j == pytest.approx(0.4, rel=0.3)
    assert st.avg_power_w == pytest.approx(4.0, rel=0.05) and st.temp_c == 40.0
    assert idle_w == pytest.approx(4.0, rel=0.05) and src == "mock-trapezoid"
