from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy import StrategyConfig, add_indicators


@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    breakout_window: int = 20
    sma_fast: int = 50
    sma_slow: int = 200


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    scfg = StrategyConfig(
        breakout_window=cfg.breakout_window,
        sma_fast=cfg.sma_fast,
        sma_slow=cfg.sma_slow,
    )
    bt = add_indicators(df, scfg)

    bt["entry_signal"] = (
        (bt["Close"] > bt["SMA_SLOW"])
        & (bt["SMA_FAST"] > bt["SMA_SLOW"])
        & (bt["Close"] > bt["HH_BREAKOUT"])
    )
    bt["position"] = bt["entry_signal"].shift(1).fillna(False).astype(int)

    bt["ret"] = bt["Close"].pct_change().fillna(0.0)
    gross = bt["position"] * bt["ret"]

    trade_change = bt["position"].diff().abs().fillna(0.0)
    fee = trade_change * (cfg.fee_bps / 10000.0)
    bt["strategy_ret"] = gross - fee

    bt["equity_curve"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["buy_hold"] = (1.0 + bt["ret"]).cumprod()

    total_return = float(bt["equity_curve"].iloc[-1] - 1.0)
    buy_hold_return = float(bt["buy_hold"].iloc[-1] - 1.0)
    max_dd = float((bt["equity_curve"] / bt["equity_curve"].cummax() - 1.0).min())

    stats = {
        "Strategy Return": total_return,
        "Buy/Hold Return": buy_hold_return,
        "Max Drawdown": max_dd,
    }
    return bt, stats
