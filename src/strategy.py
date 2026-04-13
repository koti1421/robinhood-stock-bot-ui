from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    breakout_window: int = 20
    sma_fast: int = 50
    sma_slow: int = 200
    atr_window: int = 14
    stop_atr_mult: float = 2.0


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(length).mean()


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["SMA_FAST"] = sma(out["Close"], cfg.sma_fast)
    out["SMA_SLOW"] = sma(out["Close"], cfg.sma_slow)
    out["HH_BREAKOUT"] = out["High"].rolling(cfg.breakout_window).max().shift(1)
    out["ATR"] = atr(out, cfg.atr_window)
    out["RET_63"] = out["Close"].pct_change(63)
    return out


def market_regime_on(spy_df: pd.DataFrame) -> bool:
    if len(spy_df) < 220:
        return False
    s200 = sma(spy_df["Close"], 200)
    return bool(spy_df["Close"].iloc[-1] > s200.iloc[-1])


def signal_today(df: pd.DataFrame, spy_ret_63: float, cfg: StrategyConfig) -> Dict[str, float | bool]:
    if len(df) < 220:
        return {"signal": False}

    dfi = add_indicators(df, cfg)
    row = dfi.iloc[-1]

    cond1 = row["Close"] > row["SMA_SLOW"]
    cond2 = row["SMA_FAST"] > row["SMA_SLOW"]
    cond3 = row["Close"] > row["HH_BREAKOUT"]
    cond4 = row["RET_63"] > spy_ret_63 if not np.isnan(spy_ret_63) else True

    sig = bool(cond1 and cond2 and cond3 and cond4)
    return {
        "signal": sig,
        "close": float(row["Close"]),
        "atr": float(row["ATR"]) if not np.isnan(row["ATR"]) else np.nan,
    }


def scan_universe(
    prices: Dict[str, pd.DataFrame], spy_df: pd.DataFrame, cfg: StrategyConfig
) -> List[Dict[str, float | str | bool]]:
    if not market_regime_on(spy_df):
        return []

    spy_ind = add_indicators(spy_df, cfg)
    spy_ret_63 = float(spy_ind["RET_63"].iloc[-1])

    rows: List[Dict[str, float | str | bool]] = []
    for symbol, df in prices.items():
        info = signal_today(df, spy_ret_63, cfg)
        if not info.get("signal"):
            continue
        rows.append(
            {
                "symbol": symbol,
                "close": float(info["close"]),
                "atr": float(info["atr"]),
                "signal": True,
            }
        )
    return rows
