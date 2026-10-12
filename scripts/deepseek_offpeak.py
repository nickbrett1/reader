#!/usr/bin/env python3
"""Print ``off-peak`` or ``peak`` for DeepSeek pricing at the current UTC time.

Used as a dagu precondition so the reader's concept-extraction LLM pass only
runs when DeepSeek bills at the off-peak (half) rate.

DeepSeek peak hours (api-docs.deepseek.com/quick_start/pricing, checked
2026-10-12): **01:00-04:00 and 06:00-10:00 UTC, Monday-Friday**, excluding
Chinese public holidays. Every other hour -- and all weekend -- is off-peak.

We do not model Chinese public holidays, so a holiday peak window is treated as
peak. That is conservative: we may defer a run that would in fact have been
off-peak, but we never bill at the peak rate.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

#: Half-open [start, end) hour bands, UTC, weekdays only.
PEAK_WINDOWS_UTC = ((1, 4), (6, 10))


def is_peak(now: datetime) -> bool:
    """True when DeepSeek bills at the peak rate at ``now`` (a UTC instant)."""
    if now.weekday() >= 5:  # Saturday(5) / Sunday(6): off-peak all day
        return False
    hour = now.hour + now.minute / 60.0
    return any(start <= hour < end for start, end in PEAK_WINDOWS_UTC)


def main() -> int:
    now = datetime.now(UTC)
    print("peak" if is_peak(now) else "off-peak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
