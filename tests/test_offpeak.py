"""DeepSeek off-peak gate: the window that guards the concept LLM pass."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

# scripts/ is not a package; load the module by path.
_spec = importlib.util.spec_from_file_location(
    "deepseek_offpeak", Path(__file__).resolve().parents[1] / "scripts" / "deepseek_offpeak.py"
)
offpeak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(offpeak)


def _utc(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_weekday_peak_windows_are_peak():
    # 2026-10-12 is a Monday.
    assert offpeak.is_peak(_utc(2026, 10, 12, 2)) is True
    assert offpeak.is_peak(_utc(2026, 10, 12, 7)) is True


def test_weekday_off_windows_are_off_peak():
    assert offpeak.is_peak(_utc(2026, 10, 12, 5)) is False  # between the bands
    assert offpeak.is_peak(_utc(2026, 10, 12, 12)) is False
    assert offpeak.is_peak(_utc(2026, 10, 12, 0, 30)) is False


def test_window_boundaries_are_half_open():
    assert offpeak.is_peak(_utc(2026, 10, 12, 1, 0)) is True   # start inclusive
    assert offpeak.is_peak(_utc(2026, 10, 12, 4, 0)) is False  # end exclusive
    assert offpeak.is_peak(_utc(2026, 10, 12, 6, 0)) is True
    assert offpeak.is_peak(_utc(2026, 10, 12, 10, 0)) is False


def test_weekend_is_off_peak_all_day():
    # 2026-10-10 is a Saturday, 2026-10-11 a Sunday.
    assert offpeak.is_peak(_utc(2026, 10, 10, 2)) is False
    assert offpeak.is_peak(_utc(2026, 10, 11, 7)) is False


def test_main_prints_a_verdict(capsys):
    assert offpeak.main() == 0
    assert capsys.readouterr().out.strip() in {"off-peak", "peak"}
