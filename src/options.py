from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class OptionPlan:
    symbol: str
    strategy: str
    contracts: int
    underlying_price: float
    max_loss: float
    max_profit: Optional[float]
    breakeven: Optional[float]
    risk_budget: float
    suggested_contracts: int
    valid: bool
    notes: str
    planned_at: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["planned_at"] = self.planned_at or datetime.utcnow().isoformat(timespec="seconds")
        payload["max_profit"] = "unbounded" if self.max_profit is None else float(self.max_profit)
        payload["breakeven"] = "n/a" if self.breakeven is None else float(self.breakeven)
        return payload


def _options_path() -> Path:
    base = Path(__file__).resolve().parent.parent / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "options_plans.csv"


def load_options_plans() -> pd.DataFrame:
    path = _options_path()
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "planned_at",
                "symbol",
                "strategy",
                "contracts",
                "underlying_price",
                "max_loss",
                "max_profit",
                "breakeven",
                "risk_budget",
                "suggested_contracts",
                "valid",
                "notes",
            ]
        )
    return pd.read_csv(path)


def append_options_plan(plan: OptionPlan) -> pd.DataFrame:
    df = load_options_plans()
    df = pd.concat([df, pd.DataFrame([plan.to_dict()])], ignore_index=True)
    df.to_csv(_options_path(), index=False)
    return df


def max_contracts_for_risk(max_loss_per_contract: float, risk_budget: float) -> int:
    if max_loss_per_contract <= 0:
        return 0
    return max(0, floor(risk_budget / max_loss_per_contract))


def calc_long_call(strike: float, premium: float, contracts: int) -> tuple[float, Optional[float], Optional[float], float]:
    max_loss_per_contract = premium * 100.0
    max_loss = max_loss_per_contract * contracts
    max_profit = None
    breakeven = strike + premium
    return max_loss, max_profit, breakeven, max_loss_per_contract


def calc_long_put(strike: float, premium: float, contracts: int) -> tuple[float, Optional[float], Optional[float], float]:
    max_loss_per_contract = premium * 100.0
    max_loss = max_loss_per_contract * contracts
    max_profit = (strike - premium) * 100.0 * contracts
    breakeven = strike - premium
    return max_loss, max_profit, breakeven, max_loss_per_contract


def calc_bull_call_debit(long_strike: float, short_strike: float, debit: float, contracts: int) -> tuple[float, float, float, float]:
    width = short_strike - long_strike
    max_loss_per_contract = debit * 100.0
    max_profit_per_contract = max(width - debit, 0.0) * 100.0
    return (
        max_loss_per_contract * contracts,
        max_profit_per_contract * contracts,
        long_strike + debit,
        max_loss_per_contract,
    )


def calc_bear_put_debit(long_strike: float, short_strike: float, debit: float, contracts: int) -> tuple[float, float, float, float]:
    width = long_strike - short_strike
    max_loss_per_contract = debit * 100.0
    max_profit_per_contract = max(width - debit, 0.0) * 100.0
    return (
        max_loss_per_contract * contracts,
        max_profit_per_contract * contracts,
        long_strike - debit,
        max_loss_per_contract,
    )


def calc_bull_put_credit(short_strike: float, long_strike: float, credit: float, contracts: int) -> tuple[float, float, float, float]:
    width = short_strike - long_strike
    max_loss_per_contract = max(width - credit, 0.0) * 100.0
    max_profit_per_contract = credit * 100.0
    return (
        max_loss_per_contract * contracts,
        max_profit_per_contract * contracts,
        short_strike - credit,
        max_loss_per_contract,
    )


def calc_bear_call_credit(short_strike: float, long_strike: float, credit: float, contracts: int) -> tuple[float, float, float, float]:
    width = long_strike - short_strike
    max_loss_per_contract = max(width - credit, 0.0) * 100.0
    max_profit_per_contract = credit * 100.0
    return (
        max_loss_per_contract * contracts,
        max_profit_per_contract * contracts,
        short_strike + credit,
        max_loss_per_contract,
    )
