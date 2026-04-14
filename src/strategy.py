from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    strategy_name: str = "breakout"
    breakout_window: int = 20
    sma_fast: int = 50
    sma_slow: int = 200
    atr_window: int = 14
    stop_atr_mult: float = 2.0
    rsi_window: int = 14
    mean_reversion_entry_rsi: float = 30.0


SUPPORTED_STRATEGIES = ("breakout", "pullback", "mean_reversion")


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


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["SMA_FAST"] = sma(out["Close"], cfg.sma_fast)
    out["SMA_SLOW"] = sma(out["Close"], cfg.sma_slow)
    out["HH_BREAKOUT"] = out["High"].rolling(cfg.breakout_window).max().shift(1)
    out["ATR"] = atr(out, cfg.atr_window)
    out["RSI"] = rsi(out["Close"], cfg.rsi_window)
    out["RET_63"] = out["Close"].pct_change(63)
    return out


def _signal_mask(dfi: pd.DataFrame, cfg: StrategyConfig, spy_ret_63: float | None = None) -> pd.Series:
    strategy_name = cfg.strategy_name.lower().strip()
    if strategy_name not in SUPPORTED_STRATEGIES:
        strategy_name = "breakout"

    trend_filter = (dfi["Close"] > dfi["SMA_SLOW"]) & (dfi["SMA_FAST"] > dfi["SMA_SLOW"])
    rel_strength = dfi["RET_63"] > spy_ret_63 if spy_ret_63 is not None and not np.isnan(spy_ret_63) else True

    if strategy_name == "breakout":
        entry = trend_filter & (dfi["Close"] > dfi["HH_BREAKOUT"]) & rel_strength
    elif strategy_name == "pullback":
        entry = (
            trend_filter
            & (dfi["Close"] < dfi["SMA_FAST"])
            & (dfi["Close"] > (dfi["SMA_FAST"] - 1.5 * dfi["ATR"]))
            & (dfi["RSI"] > 40)
            & rel_strength
        )
    else:
        entry = (
            (dfi["Close"] > dfi["SMA_SLOW"])
            & (dfi["Close"] < dfi["SMA_FAST"])
            & (dfi["RSI"] < cfg.mean_reversion_entry_rsi)
            & rel_strength
        )

    return entry.fillna(False)


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
    signal_series = _signal_mask(dfi, cfg, spy_ret_63=spy_ret_63)
    sig = bool(signal_series.iloc[-1])
    return {
        "signal": sig,
        "close": float(row["Close"]),
        "atr": float(row["ATR"]) if not np.isnan(row["ATR"]) else np.nan,
        "rsi": float(row["RSI"]) if not np.isnan(row["RSI"]) else np.nan,
    }


def entry_signal_series(dfi: pd.DataFrame, cfg: StrategyConfig, spy_ret_63: float | None = None) -> pd.Series:
    return _signal_mask(dfi, cfg, spy_ret_63=spy_ret_63)


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
                "rsi": float(info["rsi"]) if "rsi" in info and not np.isnan(float(info["rsi"])) else np.nan,
                "signal": True,
            }
        )
    return rows
