from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy import StrategyConfig, add_indicators, entry_signal_series


@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    strategy_name: str = "breakout"
    breakout_window: int = 20
    sma_fast: int = 50
    sma_slow: int = 200
    atr_window: int = 14
    stop_atr_mult: float = 2.0
    use_trailing_stop: bool = True


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    scfg = StrategyConfig(
        strategy_name=cfg.strategy_name,
        breakout_window=cfg.breakout_window,
        sma_fast=cfg.sma_fast,
        sma_slow=cfg.sma_slow,
        atr_window=cfg.atr_window,
        stop_atr_mult=cfg.stop_atr_mult,
    )
    bt = add_indicators(df, scfg)

    bt["entry_signal"] = entry_signal_series(bt, scfg, spy_ret_63=None)
    if cfg.use_trailing_stop:
        position = np.zeros(len(bt), dtype=int)
        trail_stop = np.full(len(bt), np.nan)
        in_position = False
        current_stop = np.nan

        for i in range(1, len(bt)):
            prev_close = float(bt["Close"].iloc[i - 1])
            prev_atr = float(bt["ATR"].iloc[i - 1]) if not pd.isna(bt["ATR"].iloc[i - 1]) else np.nan
            prev_entry_signal = bool(bt["entry_signal"].iloc[i - 1])
            exited = False

            if in_position:
                if not np.isnan(prev_atr):
                    candidate_stop = prev_close - cfg.stop_atr_mult * prev_atr
                    if np.isnan(current_stop):
                        current_stop = candidate_stop
                    else:
                        current_stop = max(current_stop, candidate_stop)

                if not np.isnan(current_stop) and prev_close <= current_stop:
                    in_position = False
                    current_stop = np.nan
                    exited = True

            if (not in_position) and (not exited) and prev_entry_signal:
                in_position = True
                if not np.isnan(prev_atr):
                    current_stop = prev_close - cfg.stop_atr_mult * prev_atr

            position[i] = 1 if in_position else 0
            if in_position and not np.isnan(current_stop):
                trail_stop[i] = current_stop

        bt["position"] = position
        bt["trail_stop"] = trail_stop
    else:
        bt["position"] = bt["entry_signal"].shift(1).fillna(False).astype(int)
        bt["trail_stop"] = np.nan

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
