from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd


JOURNAL_COLUMNS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "symbol",
    "side",
    "quantity",
    "entry_price",
    "stop_price",
    "target_price",
    "status",
    "exit_price",
    "exit_reason",
    "pnl",
    "risk_amount",
    "paper_mode",
    "strategy_tag",
    "notes",
]


@dataclass
class TradeRecord:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    risk_amount: float
    paper_mode: bool
    strategy_tag: str
    notes: str = ""
    trade_id: str = ""
    opened_at: str = ""
    closed_at: str = ""
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0

    def to_dict(self) -> dict:
        now = datetime.utcnow().isoformat(timespec="seconds")
        payload = asdict(self)
        payload["trade_id"] = self.trade_id or uuid4().hex[:12]
        payload["opened_at"] = self.opened_at or now
        payload["closed_at"] = self.closed_at or ""
        return payload


def _journal_path() -> Path:
    base = Path(__file__).resolve().parent.parent / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "trade_journal.csv"


def load_journal() -> pd.DataFrame:
    path = _journal_path()
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    df = pd.read_csv(path)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"trade_id", "opened_at", "closed_at", "symbol", "side", "status", "exit_reason", "strategy_tag", "notes"} else 0.0
    return df[JOURNAL_COLUMNS].copy()


def save_journal(df: pd.DataFrame) -> None:
    path = _journal_path()
    out = df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[JOURNAL_COLUMNS].to_csv(path, index=False)


def append_trade(record: TradeRecord) -> pd.DataFrame:
    df = load_journal()
    df = pd.concat([df, pd.DataFrame([record.to_dict()])], ignore_index=True)
    save_journal(df)
    return df


def open_positions(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    journal = load_journal() if df is None else df.copy()
    if journal.empty:
        return journal
    return journal[journal["status"] == "OPEN"].copy()


def close_position(trade_id: str, exit_price: float, exit_reason: str, notes: str = "") -> pd.DataFrame:
    df = load_journal()
    mask = (df["trade_id"] == trade_id) & (df["status"] == "OPEN")
    if not mask.any():
        raise ValueError("Trade not found or already closed.")

    idx = df[mask].index[0]
    entry_price = float(df.at[idx, "entry_price"])
    quantity = int(df.at[idx, "quantity"])
    side = str(df.at[idx, "side"]).lower()

    pnl = (exit_price - entry_price) * quantity if side == "buy" else (entry_price - exit_price) * quantity

    df.at[idx, "closed_at"] = datetime.utcnow().isoformat(timespec="seconds")
    df.at[idx, "status"] = "CLOSED"
    df.at[idx, "exit_price"] = float(exit_price)
    df.at[idx, "exit_reason"] = exit_reason
    df.at[idx, "pnl"] = float(pnl)
    if notes:
        existing = str(df.at[idx, "notes"] or "")
        df.at[idx, "notes"] = f"{existing} | {notes}".strip(" |")

    save_journal(df)
    return df


def daily_realized_pnl(df: Optional[pd.DataFrame] = None) -> float:
    journal = load_journal() if df is None else df.copy()
    if journal.empty:
        return 0.0
    closed = journal[(journal["status"] == "CLOSED") & (journal["closed_at"].astype(str) != "")].copy()
    if closed.empty:
        return 0.0
    closed["closed_day"] = pd.to_datetime(closed["closed_at"], errors="coerce").dt.date
    today = datetime.utcnow().date()
    pnl = pd.to_numeric(closed.loc[closed["closed_day"] == today, "pnl"], errors="coerce").fillna(0.0)
    return float(pnl.sum())


def summarize_journal(df: Optional[pd.DataFrame] = None) -> dict:
    journal = load_journal() if df is None else df.copy()
    if journal.empty:
        return {
            "Open Positions": 0,
            "Closed Trades": 0,
            "Win Rate": 0.0,
            "Net PnL": 0.0,
            "Avg Winner": 0.0,
            "Avg Loser": 0.0,
            "Expectancy": 0.0,
        }

    closed = journal[journal["status"] == "CLOSED"].copy()
    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]
    closed_count = len(closed)
    win_rate = float(len(wins) / closed_count) if closed_count else 0.0
    avg_winner = float(wins["pnl"].mean()) if not wins.empty else 0.0
    avg_loser = float(losses["pnl"].mean()) if not losses.empty else 0.0
    expectancy = win_rate * avg_winner + (1.0 - win_rate) * avg_loser

    return {
        "Open Positions": int((journal["status"] == "OPEN").sum()),
        "Closed Trades": int(closed_count),
        "Win Rate": win_rate,
        "Net PnL": float(closed["pnl"].sum()) if closed_count else 0.0,
        "Avg Winner": avg_winner,
        "Avg Loser": avg_loser,
        "Expectancy": float(expectancy),
    }