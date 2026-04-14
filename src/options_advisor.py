from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class OptionsRecommendation:
    strategy: str
    reason: str
    risk_profile: str
    ideal_iv: str
    probability_description: str
    example: str
    priority: int  # 1 = best fit


def _historical_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def iv_rank_proxy(df: pd.DataFrame) -> Dict[str, float]:
    """Use historical volatility percentile as IV rank proxy."""
    if len(df) < 252:
        return {"iv_rank": 50.0, "current_hv": 0.0, "hv_1y_high": 0.0, "hv_1y_low": 0.0}

    hv = _historical_volatility(df, 21).dropna()
    if hv.empty:
        return {"iv_rank": 50.0, "current_hv": 0.0, "hv_1y_high": 0.0, "hv_1y_low": 0.0}

    current = float(hv.iloc[-1])
    last_252 = hv.tail(252)
    high = float(last_252.max())
    low = float(last_252.min())

    if high == low:
        rank = 50.0
    else:
        rank = float((current - low) / (high - low) * 100)

    return {
        "iv_rank": round(rank, 1),
        "current_hv": round(current, 1),
        "hv_1y_high": round(high, 1),
        "hv_1y_low": round(low, 1),
    }


def recommend_strategies(
    outlook: str,
    iv_rank: float,
    risk_tolerance: str = "moderate",
) -> List[OptionsRecommendation]:
    """
    Decision tree for options strategy selection.

    outlook: "bullish", "bearish", "neutral", "volatile"
    iv_rank: 0-100
    risk_tolerance: "conservative", "moderate", "aggressive"
    """
    recs: List[OptionsRecommendation] = []
    high_iv = iv_rank >= 50
    low_iv = iv_rank < 50

    if outlook == "bullish":
        if low_iv:
            recs.append(OptionsRecommendation(
                strategy="Long Call",
                reason="Low IV = cheap premiums. Directional play on upside.",
                risk_profile="Max loss = premium paid. Unlimited upside.",
                ideal_iv="Low IV (buy cheap options)",
                probability_description="Lower probability but high reward potential when right.",
                example="Buy ATM or slightly OTM call 30-60 DTE. Risk 1-2% of account.",
                priority=1,
            ))
            recs.append(OptionsRecommendation(
                strategy="Bull Call Debit Spread",
                reason="Low IV. Reduces cost vs naked long call with capped upside.",
                risk_profile="Max loss = debit paid. Max profit = spread width - debit.",
                ideal_iv="Low IV (buy spreads cheap)",
                probability_description="Moderate probability. Defined risk/reward.",
                example="Buy 1 ATM call, sell 1 OTM call $5-10 wide. 30-45 DTE.",
                priority=2,
            ))
        else:
            recs.append(OptionsRecommendation(
                strategy="Bull Put Credit Spread",
                reason="High IV = fat premiums. Sell premium with bullish bias.",
                risk_profile="Max loss = spread width - credit. Max profit = credit received.",
                ideal_iv="High IV (sell expensive premium)",
                probability_description="High probability (~65-80%) if sold OTM.",
                example="Sell OTM put, buy lower put for protection. 30-45 DTE.",
                priority=1,
            ))
            recs.append(OptionsRecommendation(
                strategy="Bull Call Debit Spread",
                reason="Still directional but IV drag makes naked calls expensive.",
                risk_profile="Max loss = debit. Spread reduces IV risk.",
                ideal_iv="Any IV (spread neutralizes vega)",
                probability_description="Moderate probability with defined risk.",
                example="Buy ATM call, sell OTM call. Keep spread tight to manage cost.",
                priority=2,
            ))

    elif outlook == "bearish":
        if low_iv:
            recs.append(OptionsRecommendation(
                strategy="Long Put",
                reason="Low IV makes puts cheap. Direct downside exposure.",
                risk_profile="Max loss = premium paid. Profit as stock falls.",
                ideal_iv="Low IV (buy cheap protection)",
                probability_description="Lower probability but strong payoff in selloffs.",
                example="Buy ATM or slightly OTM put 30-60 DTE.",
                priority=1,
            ))
            recs.append(OptionsRecommendation(
                strategy="Bear Put Debit Spread",
                reason="Cheaper than naked puts. Defined risk/reward.",
                risk_profile="Max loss = debit paid. Max profit = spread width - debit.",
                ideal_iv="Low IV",
                probability_description="Moderate probability with defined risk.",
                example="Buy ATM put, sell OTM put $5-10 below. 30-45 DTE.",
                priority=2,
            ))
        else:
            recs.append(OptionsRecommendation(
                strategy="Bear Call Credit Spread",
                reason="High IV = collect rich premium with bearish bias.",
                risk_profile="Max loss = spread width - credit. Max profit = credit.",
                ideal_iv="High IV (sell expensive calls)",
                probability_description="High probability (~65-80%) if sold OTM.",
                example="Sell OTM call, buy higher call. 30-45 DTE.",
                priority=1,
            ))

    elif outlook == "neutral":
        if high_iv:
            recs.append(OptionsRecommendation(
                strategy="Iron Condor (sell both spreads)",
                reason="High IV + no direction = sell premium on both sides.",
                risk_profile="Max loss = wider spread width - credits. Max profit = total credit.",
                ideal_iv="High IV (collect maximum time decay)",
                probability_description="High probability (~70-85%) with narrow range.",
                example="Sell OTM put spread + OTM call spread. 30-45 DTE.",
                priority=1,
            ))
            recs.append(OptionsRecommendation(
                strategy="Bull Put Credit Spread",
                reason="If leaning slightly bullish within neutral, sell puts.",
                risk_profile="Max loss = width - credit. Max profit = credit.",
                ideal_iv="High IV",
                probability_description="High probability when sold OTM.",
                example="Sell put spread below support. 30-45 DTE.",
                priority=2,
            ))
        else:
            recs.append(OptionsRecommendation(
                strategy="Calendar Spread",
                reason="Low IV + neutral = buy back-month, sell front-month.",
                risk_profile="Max loss = debit paid. Profits from time decay differential.",
                ideal_iv="Low IV (anticipate IV expansion)",
                probability_description="Moderate probability. Best near earnings/events.",
                example="Sell near-term ATM, buy same-strike further out.",
                priority=1,
            ))

    else:  # volatile
        if low_iv:
            recs.append(OptionsRecommendation(
                strategy="Long Straddle",
                reason="Expect big move, IV is cheap. Buy both call and put.",
                risk_profile="Max loss = total premium. Profits from large move either way.",
                ideal_iv="Low IV (buy before volatility expansion)",
                probability_description="Lower probability but huge payoff if move is large enough.",
                example="Buy ATM call + ATM put, same strike, 30-45 DTE.",
                priority=1,
            ))
        else:
            recs.append(OptionsRecommendation(
                strategy="Long Strangle",
                reason="Expect big move but IV is elevated. OTM options reduce cost.",
                risk_profile="Max loss = total premium. Profits from large move.",
                ideal_iv="Moderate-High IV (OTM reduces cost)",
                probability_description="Lower probability. Need significant price movement.",
                example="Buy OTM call + OTM put. 30-45 DTE.",
                priority=1,
            ))

    return sorted(recs, key=lambda r: r.priority)


def options_decision_summary(
    symbol: str,
    df: pd.DataFrame,
    outlook: str,
) -> Dict[str, object]:
    """Full options analysis for a symbol."""
    iv_data = iv_rank_proxy(df)
    strategies = recommend_strategies(outlook, iv_data["iv_rank"])

    close = float(df["Close"].iloc[-1]) if not df.empty else 0.0

    ret_5d = float(df["Close"].pct_change(5).iloc[-1] * 100) if len(df) > 5 else 0
    ret_21d = float(df["Close"].pct_change(21).iloc[-1] * 100) if len(df) > 21 else 0

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "iv_rank": iv_data["iv_rank"],
        "current_hv": iv_data["current_hv"],
        "hv_1y_high": iv_data["hv_1y_high"],
        "hv_1y_low": iv_data["hv_1y_low"],
        "ret_5d": round(ret_5d, 2),
        "ret_21d": round(ret_21d, 2),
        "outlook": outlook,
        "recommendations": strategies,
    }


# --- Key rules for successful options trading ---
RULES = [
    "Never risk more than 1-2% of account on a single options trade",
    "Only trade defined-risk strategies (spreads) until consistently profitable",
    "Sell premium when IV rank is HIGH (>50), buy premium when LOW (<30)",
    "Target 30-45 DTE for optimal theta decay balance",
    "Close winners at 50% max profit — don't let greed erase gains",
    "Close losers at 2x the credit received or when thesis breaks",
    "Avoid holding through earnings unless that IS the thesis",
    "Check bid/ask spread — if >10% of mid price, liquidity is poor",
    "Size positions so max loss per trade stays within your risk budget",
    "Track every trade: entry reason, exit plan, actual result, and lesson learned",
]
