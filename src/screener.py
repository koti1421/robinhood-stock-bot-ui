from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StockScore:
    symbol: str
    momentum: float        # 0-100
    trend_quality: float   # 0-100
    relative_strength: float  # 0-100
    volatility_quality: float  # 0-100
    volume_trend: float    # 0-100
    composite: float       # weighted 0-100
    grade: str             # A/B/C/D/F
    verdict: str           # "Strong Buy" / "Buy" / "Watch" / "Avoid"
    close: float
    atr: float
    rsi: float
    sma_trend: str         # "Bullish" / "Bearish" / "Sideways"


def _pct_rank(value: float, series: pd.Series) -> float:
    """Percentile rank of value within series (0-100)."""
    if series.std() == 0:
        return 50.0
    return float((series < value).sum() / max(len(series), 1) * 100.0)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _momentum_score(df: pd.DataFrame) -> float:
    if len(df) < 63:
        return 0.0
    ret_21 = float(df["Close"].pct_change(21).iloc[-1]) * 100
    ret_63 = float(df["Close"].pct_change(63).iloc[-1]) * 100
    ret_126 = float(df["Close"].pct_change(min(126, len(df) - 1)).iloc[-1]) * 100 if len(df) > 126 else ret_63

    raw = 0.20 * ret_21 + 0.45 * ret_63 + 0.35 * ret_126
    return _clamp(50 + raw * 2.5, 0, 100)


def _trend_quality_score(df: pd.DataFrame, sma_fast: int = 50, sma_slow: int = 200) -> tuple[float, str]:
    if len(df) < sma_slow + 10:
        return 0.0, "Sideways"

    close = df["Close"].iloc[-1]
    fast = df["Close"].rolling(sma_fast).mean().iloc[-1]
    slow = df["Close"].rolling(sma_slow).mean().iloc[-1]

    fast_slope = (df["Close"].rolling(sma_fast).mean().iloc[-1] - df["Close"].rolling(sma_fast).mean().iloc[-21]) / df["Close"].rolling(sma_fast).mean().iloc[-21] * 100 if len(df) > sma_fast + 21 else 0

    above_fast = close > fast
    above_slow = close > slow
    fast_above_slow = fast > slow

    if above_fast and above_slow and fast_above_slow:
        trend = "Bullish"
        base = 75
    elif above_slow and fast_above_slow:
        trend = "Bullish"
        base = 60
    elif above_slow:
        trend = "Sideways"
        base = 45
    else:
        trend = "Bearish"
        base = 20

    score = base + fast_slope * 2
    return _clamp(score, 0, 100), trend


def _relative_strength_score(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    if len(stock_df) < 63 or len(spy_df) < 63:
        return 50.0

    stock_ret = float(stock_df["Close"].pct_change(63).iloc[-1])
    spy_ret = float(spy_df["Close"].pct_change(63).iloc[-1])
    diff = (stock_ret - spy_ret) * 100

    return _clamp(50 + diff * 3, 0, 100)


def _volatility_quality_score(df: pd.DataFrame, atr_window: int = 14) -> float:
    if len(df) < atr_window + 20:
        return 50.0

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(atr_window).mean()
    atr_pct = (atr_series / close * 100).dropna()

    if atr_pct.empty:
        return 50.0

    current_atr_pct = float(atr_pct.iloc[-1])
    hist_median = float(atr_pct.median())

    if current_atr_pct < hist_median * 0.8:
        return 80.0
    elif current_atr_pct < hist_median:
        return 65.0
    elif current_atr_pct < hist_median * 1.3:
        return 50.0
    else:
        return 30.0


def _volume_trend_score(df: pd.DataFrame) -> float:
    if "Volume" not in df.columns or len(df) < 50:
        return 50.0

    vol = df["Volume"].copy()
    if vol.sum() == 0:
        return 50.0

    avg_20 = vol.rolling(20).mean().iloc[-1]
    avg_50 = vol.rolling(50).mean().iloc[-1]

    if avg_50 == 0:
        return 50.0

    ratio = avg_20 / avg_50
    rising_price = df["Close"].iloc[-1] > df["Close"].iloc[-20]

    if ratio > 1.2 and rising_price:
        return 80.0
    elif ratio > 1.0 and rising_price:
        return 65.0
    elif ratio < 0.8 and not rising_price:
        return 30.0
    else:
        return 50.0


def _rsi(series: pd.Series, length: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi_vals.iloc[-1]) if not rsi_vals.empty else 50.0


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 35:
        return "D"
    return "F"


def _verdict(score: float, trend: str) -> str:
    if score >= 75 and trend == "Bullish":
        return "Strong Buy"
    elif score >= 60 and trend in ("Bullish", "Sideways"):
        return "Buy"
    elif score >= 45:
        return "Watch"
    return "Avoid"


def score_stock(
    symbol: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sma_fast: int = 50,
    sma_slow: int = 200,
    atr_window: int = 14,
) -> Optional[StockScore]:
    if df.empty or len(df) < 60:
        return None

    mom = _momentum_score(df)
    trend_q, trend_label = _trend_quality_score(df, sma_fast, sma_slow)
    rel = _relative_strength_score(df, spy_df)
    vol_q = _volatility_quality_score(df, atr_window)
    vol_t = _volume_trend_score(df)

    composite = (
        0.30 * mom +
        0.25 * trend_q +
        0.20 * rel +
        0.15 * vol_q +
        0.10 * vol_t
    )

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr_val = float(tr.rolling(atr_window).mean().iloc[-1])
    rsi_val = _rsi(df["Close"])

    return StockScore(
        symbol=symbol,
        momentum=round(mom, 1),
        trend_quality=round(trend_q, 1),
        relative_strength=round(rel, 1),
        volatility_quality=round(vol_q, 1),
        volume_trend=round(vol_t, 1),
        composite=round(composite, 1),
        grade=_grade(composite),
        verdict=_verdict(composite, trend_label),
        close=round(float(df["Close"].iloc[-1]), 2),
        atr=round(atr_val, 2),
        rsi=round(rsi_val, 1),
        sma_trend=trend_label,
    )


def rank_universe(
    prices: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    sma_fast: int = 50,
    sma_slow: int = 200,
    atr_window: int = 14,
) -> List[StockScore]:
    scores = []
    for symbol, df in prices.items():
        s = score_stock(symbol, df, spy_df, sma_fast, sma_slow, atr_window)
        if s is not None:
            scores.append(s)
    scores.sort(key=lambda x: x.composite, reverse=True)
    return scores


def detect_market_regime(spy_df: pd.DataFrame) -> Dict[str, object]:
    if len(spy_df) < 220:
        return {"regime": "Unknown", "strength": 0, "description": "Insufficient data"}

    close = spy_df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    current = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1])

    above_200 = current > s200
    above_50 = current > s50
    golden = s50 > s200

    ret_21 = float(close.pct_change(21).iloc[-1]) * 100
    ret_63 = float(close.pct_change(63).iloc[-1]) * 100

    high = close.rolling(252).max()
    drawdown = float((current / high.iloc[-1] - 1) * 100)

    if above_200 and golden and ret_63 > 0:
        regime = "Strong Uptrend"
        strength = 90
        desc = "SPY above rising 50 & 200 SMA. Ideal for trend-following longs."
        color = "#00E676"
    elif above_200 and golden:
        regime = "Uptrend"
        strength = 70
        desc = "SPY above 200 SMA with golden cross. Healthy for longs with normal sizing."
        color = "#4CAF50"
    elif above_200:
        regime = "Cautious Bull"
        strength = 50
        desc = "SPY above 200 SMA but 50 SMA weakening. Reduce position sizes."
        color = "#FFC107"
    elif drawdown > -10:
        regime = "Pullback"
        strength = 35
        desc = "SPY below key averages but shallow drawdown. Wait for stabilization."
        color = "#FF9800"
    elif drawdown > -20:
        regime = "Correction"
        strength = 20
        desc = "Market in correction territory. Avoid new longs or hedge."
        color = "#F44336"
    else:
        regime = "Bear Market"
        strength = 5
        desc = "Severe drawdown. Capital preservation mode."
        color = "#D32F2F"

    return {
        "regime": regime,
        "strength": strength,
        "description": desc,
        "color": color,
        "spy_price": round(current, 2),
        "sma50": round(s50, 2),
        "sma200": round(s200, 2),
        "ret_21d": round(ret_21, 2),
        "ret_63d": round(ret_63, 2),
        "drawdown": round(drawdown, 2),
    }
