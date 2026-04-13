from __future__ import annotations

import math


def position_size(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    max_notional_pct: float = 0.10,
) -> int:
    per_share_risk = max(entry_price - stop_price, 0.01)
    raw_shares = math.floor((equity * risk_per_trade) / per_share_risk)
    max_notional_shares = math.floor((equity * max_notional_pct) / entry_price)
    return max(0, min(raw_shares, max_notional_shares))
