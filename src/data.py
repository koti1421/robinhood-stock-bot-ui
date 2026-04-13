from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    out = df[cols].dropna().copy()
    out.index = pd.to_datetime(out.index)
    return out


def fetch_prices(symbol: str, years: int = 3) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    return _normalize(raw)


def fetch_many(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = fetch_prices(symbol, years=years)
        if not df.empty:
            data[symbol] = df
    return data
