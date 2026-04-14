from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.backtest import BacktestConfig, run_backtest
from src.data import fetch_many, fetch_prices
from src.journal import (
    TradeRecord, append_trade, close_position,
    daily_realized_pnl, load_journal, open_positions, summarize_journal,
)
from src.options import (
    OptionPlan, append_options_plan, calc_bear_call_credit, calc_bear_put_debit,
    calc_bull_call_debit, calc_bull_put_credit, calc_long_call, calc_long_put,
    load_options_plans, max_contracts_for_risk,
)
from src.options_advisor import (
    RULES as OPTIONS_RULES, iv_rank_proxy, options_decision_summary, recommend_strategies,
)
from src.risk import position_size
from src.robinhood_client import RobinhoodClient, RobinhoodOrderRequest, env_credentials
from src.screener import StockScore, detect_market_regime, rank_universe, score_stock
from src.strategy import (
    StrategyConfig, add_indicators, market_regime_on, scan_universe,
)

STRATEGY_LABELS = {
    "breakout": "Trend Breakout",
    "pullback": "Trend Pullback",
    "mean_reversion": "Mean Reversion",
}

load_dotenv()

st.set_page_config(
    page_title="ProTrader Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0b1120;
    --bg-secondary: #111827;
    --bg-card: #152036;
    --bg-card-hover: #1c2d4a;
    --border: rgba(148, 210, 255, 0.15);
    --border-glow: rgba(148, 210, 255, 0.35);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #a1b0c8;
    --accent-blue: #60a5fa;
    --accent-cyan: #22d3ee;
    --accent-green: #34d399;
    --accent-red: #f87171;
    --accent-amber: #fbbf24;
    --accent-purple: #a78bfa;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Force all text bright */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-primary);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* ALL headings bright white */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Captions / small text */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1629 0%, #162240 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4 {
    color: var(--accent-cyan) !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] .stSlider p,
section[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text-primary) !important;
}

/* Input fields */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}

/* Checkbox labels bright */
.stCheckbox label span {
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0c1b3a 0%, #1a3a6b 40%, #0e4f7a 100%);
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.1), transparent 70%);
    pointer-events: none;
}

.hero-banner h2 {
    color: #ffffff !important;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.15;
}

.hero-banner p {
    color: rgba(255,255,255,0.85) !important;
    margin: 8px 0 0;
    font-size: 1rem;
}

.hero-chip {
    display: inline-block;
    margin-top: 14px;
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.chip-paper {
    background: rgba(251, 191, 36, 0.2);
    border: 1px solid rgba(251, 191, 36, 0.5);
    color: #fde68a !important;
}

.chip-live {
    background: rgba(52, 211, 153, 0.2);
    border: 1px solid rgba(52, 211, 153, 0.5);
    color: #6ee7b7 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 14px;
    transition: border-color 0.2s;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--border-glow);
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
}

/* Glass Card */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 0.8rem;
    color: var(--text-primary);
}

.glass-card * { color: inherit; }

.glass-card-glow {
    background: linear-gradient(135deg, rgba(96, 165, 250, 0.08) 0%, var(--bg-card) 100%);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 0.8rem;
    color: var(--text-primary);
}

.glass-card-glow * { color: inherit; }

/* Regime indicator - HIGH CONTRAST */
.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    color: #ffffff !important;
}

/* Grade badges */
.grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
}

.grade-A { background: rgba(52,211,153,0.2); color: #6ee7b7 !important; border: 1px solid rgba(52,211,153,0.4); }
.grade-B { background: rgba(96,165,250,0.2); color: #93bbfd !important; border: 1px solid rgba(96,165,250,0.4); }
.grade-C { background: rgba(251,191,36,0.2); color: #fde68a !important; border: 1px solid rgba(251,191,36,0.4); }
.grade-D { background: rgba(251,146,60,0.2); color: #fdba74 !important; border: 1px solid rgba(251,146,60,0.4); }
.grade-F { background: rgba(248,113,113,0.2); color: #fca5a5 !important; border: 1px solid rgba(248,113,113,0.4); }

/* Score pill */
.verdict-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.verdict-strong-buy { background: rgba(52,211,153,0.2); color: #6ee7b7 !important; border: 1px solid rgba(52,211,153,0.4); }
.verdict-buy { background: rgba(96,165,250,0.2); color: #93bbfd !important; border: 1px solid rgba(96,165,250,0.4); }
.verdict-watch { background: rgba(251,191,36,0.2); color: #fde68a !important; border: 1px solid rgba(251,191,36,0.4); }
.verdict-avoid { background: rgba(248,113,113,0.2); color: #fca5a5 !important; border: 1px solid rgba(248,113,113,0.4); }

/* IV Gauge */
.iv-bar-container {
    background: rgba(30, 41, 59, 0.9);
    border-radius: 8px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
}
.iv-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* Strategy card */
.strat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    color: var(--text-primary);
}
.strat-card * { color: inherit; }

.strat-card-best {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, var(--bg-card) 100%);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    color: var(--text-primary);
}
.strat-card-best * { color: inherit; }

/* Rules */
.rule-item {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-cyan);
    padding: 10px 14px;
    margin-bottom: 6px;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: var(--text-primary) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    color: var(--text-secondary) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
}

/* Status row - HIGH CONTRAST */
.status-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 0.75rem;
    color: var(--text-primary) !important;
    font-size: 0.92rem;
}

.status-row strong { color: #ffffff !important; }

/* Recommendation banner */
.recommend {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, var(--bg-card) 100%);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 0.5rem 0 0.9rem;
    color: var(--text-primary) !important;
}
.recommend strong { color: var(--accent-green) !important; }

/* Plotly chart backgrounds */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* Score bar */
.score-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 3px 0;
}
.score-bar-label {
    width: 100px;
    font-size: 0.8rem;
    color: var(--text-muted) !important;
    text-align: right;
}
.score-bar-track {
    flex: 1;
    height: 7px;
    background: rgba(30,41,59,0.9);
    border-radius: 4px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.score-bar-value {
    width: 36px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary) !important;
    font-weight: 500;
}

/* Expander text */
.stExpander, .stExpander p, .stExpander span, .stExpander div {
    color: var(--text-primary) !important;
}

/* Table / dataframe text */
.stDataFrame, [data-testid="stDataFrame"] {
    color: var(--text-primary) !important;
}

/* Info/Warning/Error boxes - improve readability */
.stAlert p, .stAlert span {
    font-size: 0.92rem !important;
}

/* Markdown in custom HTML - force readable colors */
.glass-card span[style*="color:#a1b0c8"],
.glass-card-glow span[style*="color:#a1b0c8"],
.strat-card span[style*="color:#a1b0c8"],
.strat-card-best span[style*="color:#a1b0c8"] {
    color: #a1b0c8 !important;
}

.glass-card span[style*="color:#cbd5e1"],
.glass-card-glow span[style*="color:#cbd5e1"],
.strat-card span[style*="color:#cbd5e1"],
.strat-card-best span[style*="color:#cbd5e1"] {
    color: #cbd5e1 !important;
}

.glass-card span[style*="color:#f1f5f9"],
.glass-card-glow span[style*="color:#f1f5f9"],
.strat-card span[style*="color:#f1f5f9"],
.strat-card-best span[style*="color:#f1f5f9"] {
    color: #f1f5f9 !important;
}

/* ─── Selectbox / Dropdown - FORCE BRIGHT TEXT ─── */
[data-baseweb="select"] {
    background: #1e2d4a !important;
    border: 1px solid rgba(148, 210, 255, 0.3) !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] * {
    color: #ffffff !important;
    font-weight: 600 !important;
}

[data-baseweb="select"] div[value] {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

[data-baseweb="select"] input {
    color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Dropdown menu list */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #1e2d4a !important;
    border: 1px solid rgba(148, 210, 255, 0.3) !important;
}

ul[role="listbox"] li,
li[role="option"] {
    color: #ffffff !important;
    font-weight: 500 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: #2a3f66 !important;
    color: #ffffff !important;
}

/* Multiselect tags */
[data-baseweb="tag"] {
    background: rgba(96, 165, 250, 0.2) !important;
    color: #f1f5f9 !important;
}

/* Radio buttons & selectbox labels in sidebar */
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #f1f5f9 !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #f1f5f9 !important;
}

/* Slider value text */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-baseweb="slider"] div {
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Functions ───────────────────────────────────────────────────────────────

def parse_symbols(raw_symbols: str) -> List[str]:
    cleaned = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    deduped: List[str] = []
    for sym in cleaned:
        if re.fullmatch(r"[A-Z\.\-]{1,8}", sym) and sym not in deduped:
            deduped.append(sym)
    return deduped


def build_position_manager(open_df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    if open_df.empty:
        return pd.DataFrame()

    prices = fetch_many(open_df["symbol"].astype(str).str.upper().tolist(), years=1)
    rows = []
    for _, trade in open_df.iterrows():
        symbol = str(trade["symbol"]).upper()
        df = prices.get(symbol)
        if df is None or df.empty:
            continue

        ind = add_indicators(df, cfg)
        latest = ind.iloc[-1]
        current = float(latest["Close"])
        atr_now = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else np.nan
        entry = float(trade["entry_price"])
        stop = float(trade["stop_price"])
        target = float(trade["target_price"])
        qty = int(trade["quantity"])
        risk_per_share = max(entry - stop, 0.01)
        partial_target = entry + risk_per_share
        trailing_stop = max(stop, current - cfg.stop_atr_mult * atr_now) if not np.isnan(atr_now) else stop
        unrealized = (current - entry) * qty
        r_multiple = (current - entry) / risk_per_share

        if current <= stop:
            action = "🔴 Exit: stop hit"
        elif current >= target:
            action = "🟢 Take profit: 2R"
        elif current >= partial_target:
            action = f"🟡 Scale out, trail ${trailing_stop:.2f}"
        else:
            action = "⚪ Hold"

        rows.append({
            "Trade ID": trade["trade_id"],
            "Symbol": symbol,
            "Qty": qty,
            "Entry": round(entry, 2),
            "Current": round(current, 2),
            "Stop": round(stop, 2),
            "Partial 1R": round(partial_target, 2),
            "Target 2R": round(target, 2),
            "Trail Stop": round(trailing_stop, 2),
            "Unrealized PnL": round(unrealized, 2),
            "R": round(r_multiple, 2),
            "Action": action,
        })
    return pd.DataFrame(rows)


def score_bar_html(label: str, value: float, color: str) -> str:
    return f"""<div class="score-bar">
        <span class="score-bar-label">{label}</span>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:{value}%;background:{color};"></div></div>
        <span class="score-bar-value">{value:.0f}</span>
    </div>"""


def grade_color(grade: str) -> str:
    return {"A": "#10b981", "B": "#3b82f6", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}.get(grade, "#64748b")


def verdict_class(v: str) -> str:
    return "verdict-" + v.lower().replace(" ", "-")


def dark_plotly_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(26,35,50,0.9)",
        plot_bgcolor="rgba(10,14,23,0.95)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="rgba(99,179,237,0.06)"),
        yaxis=dict(gridcolor="rgba(99,179,237,0.06)"),
    )
    return fig


# ─── Config ──────────────────────────────────────────────────────────────────

paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
mode_badge = "PAPER MODE" if paper_mode else "LIVE MODE"
chip_class = "chip-paper" if paper_mode else "chip-live"

st.markdown(
    f"""<div class="hero-banner">
    <h2>ProTrader Dashboard</h2>
    <p>Smart stock scoring · Options strategy advisor · Risk-first execution</p>
    <span class="hero-chip {chip_class}">{mode_badge}</span>
</div>""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("#### 🌐 Universe")
    symbols_input = st.text_area(
        "Symbols (comma-separated)",
        value="AMD,SOFI,PLTR,BAC,F,INTC,HOOD,COIN,SQ,UBER,SNAP,RIVN,AAPL,META,NVDA",
        height=92,
    )

    st.markdown("#### 💰 Risk Controls")
    equity = st.number_input("Account Equity ($)", value=500.0, min_value=100.0, step=100.0)
    risk_per_trade = st.slider("Risk Per Trade", min_value=0.005, max_value=0.05, value=0.02, step=0.005, format="%.3f")
    daily_loss_limit_pct = st.slider("Daily Loss Lockout", min_value=0.005, max_value=0.05, value=0.02, step=0.005, format="%.3f")
    max_open_positions = st.number_input("Max Open Positions", min_value=1, max_value=20, value=2, step=1)

    st.markdown("#### ⚙️ Strategy Engine")
    strategy_name = st.selectbox(
        "Strategy Type",
        options=["breakout", "pullback", "mean_reversion"],
        format_func=lambda x: STRATEGY_LABELS.get(x, x),
    )
    breakout_window = st.slider("Breakout Window", min_value=10, max_value=80, value=20, step=5)
    sma_fast = st.slider("Fast SMA", min_value=20, max_value=120, value=50, step=5)
    sma_slow = st.slider("Slow SMA", min_value=100, max_value=300, value=200, step=10)
    atr_window = st.slider("ATR Window", min_value=7, max_value=30, value=14)
    stop_atr_mult = st.slider("Stop ATR Mult", min_value=1.0, max_value=4.0, value=2.0, step=0.25)

# ─── Validate ────────────────────────────────────────────────────────────────

symbols = parse_symbols(symbols_input)
if not symbols:
    st.error("Add at least one valid symbol.")
    st.stop()
if sma_fast >= sma_slow:
    st.error("Fast SMA must be lower than Slow SMA.")
    st.stop()

strategy_cfg = StrategyConfig(
    strategy_name=strategy_name,
    breakout_window=breakout_window,
    sma_fast=sma_fast,
    sma_slow=sma_slow,
    atr_window=atr_window,
    stop_atr_mult=stop_atr_mult,
)

# ─── Load state  ─────────────────────────────────────────────────────────────

journal_df = load_journal()
open_df = open_positions(journal_df)
journal_summary = summarize_journal(journal_df)
today_realized_pnl = daily_realized_pnl(journal_df)
daily_loss_limit_value = equity * daily_loss_limit_pct
daily_lockout = today_realized_pnl <= -daily_loss_limit_value
available_slots = max(int(max_open_positions) - len(open_df), 0)

spy_df = fetch_prices("SPY", years=3)
regime = market_regime_on(spy_df)
regime_info = detect_market_regime(spy_df)

# ─── Top Metrics Row ─────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Market Regime", regime_info["regime"])
c2.metric("SPY", f"${regime_info['spy_price']:,.2f}")
c3.metric("Universe", f"{len(symbols)} stocks")
c4.metric("Risk/Trade", f"{risk_per_trade*100:.2f}%")
c5.metric("Positions", f"{len(open_df)}/{int(max_open_positions)}")

st.markdown(
    f"""<div class="regime-badge" style="background:rgba({','.join(str(int(regime_info['color'].lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12); border:1px solid {regime_info['color']}40; color:{regime_info['color']};">
    ● {regime_info['regime']} — {regime_info['description']}</div>""",
    unsafe_allow_html=True,
)

if daily_lockout:
    st.error(f"Daily loss lockout active. Realized PnL today: ${today_realized_pnl:,.2f} (limit: -${daily_loss_limit_value:,.2f}). New buys blocked.")
else:
    st.markdown(
        f"<div class='status-row'><strong>Risk:</strong> Realized today ${today_realized_pnl:,.2f} · Slots {available_slots} · Lockout at -${daily_loss_limit_value:,.2f}</div>",
        unsafe_allow_html=True,
    )

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌅 Morning Routine",
    "📊 Stock Screener",
    "🎯 Signal Scanner",
    "📈 Backtest",
    "🔗 Robinhood",
    "📓 Journal",
    "🎰 Options Advisor",
    "📺 TradingView Analysis",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — MORNING ROUTINE
# ═════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("### 🌅 Morning Trading Routine")
    st.caption("Follow this checklist every morning before the market opens at 9:30 AM ET.")

    risk_budget_per_trade = equity * risk_per_trade

    st.markdown(f"""<div class="glass-card-glow">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.75rem; color:#a1b0c8; text-transform:uppercase; letter-spacing:0.5px;">Today's Trading Capital</div>
                <div style="font-size:2rem; font-weight:800; color:#22d3ee; font-family:'JetBrains Mono',monospace;">${equity:,.0f}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.75rem; color:#a1b0c8;">Risk per Trade</div>
                <div style="font-size:1.4rem; font-weight:700; color:#fbbf24; font-family:'JetBrains Mono',monospace;">${risk_budget_per_trade:,.2f}</div>
                <div style="font-size:0.75rem; color:#a1b0c8;">({risk_per_trade*100:.1f}% of equity)</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Step-by-Step Workflow")

    step1 = st.checkbox("**Step 1: Check Market Regime** — Is SPY above 200 SMA?", key="mr_step1")
    if step1:
        regime_ok = regime_info["regime"] in ("Strong Uptrend", "Uptrend")
        if regime_ok:
            st.success(f"✅ Market regime: **{regime_info['regime']}** — Good to trade longs.")
        elif regime_info["regime"] == "Cautious Bull":
            st.warning(f"⚠️ Market regime: **{regime_info['regime']}** — Trade with half size.")
        else:
            st.error(f"🚫 Market regime: **{regime_info['regime']}** — Avoid new longs. Stay in cash.")

    step2 = st.checkbox("**Step 2: Run Stock Screener** — Find A and B grade stocks", key="mr_step2")
    if step2:
        st.info("Go to the **Stock Screener** tab and click **Run Screener**. Focus only on A and B grades.")

    step3 = st.checkbox("**Step 3: Run Signal Scanner** — Check for entry signals", key="mr_step3")
    if step3:
        st.info("Go to the **Signal Scanner** tab and click **Scan Now**. Only trade stocks that are BOTH high-grade AND have signals.")

    step4 = st.checkbox("**Step 4: Calculate Position Size** — Use the sizing rules", key="mr_step4")
    if step4:
        st.markdown(f"""<div class="glass-card">
            <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7;">
                <strong style="color:#22d3ee;">Position Sizing Formula:</strong><br>
                • Entry price from scanner<br>
                • Stop = Entry − ({stop_atr_mult}× ATR)<br>
                • Risk per share = Entry − Stop<br>
                • Max shares = ${risk_budget_per_trade:.2f} ÷ Risk per share<br>
                • Max position = {10}% of equity = ${equity * 0.10:,.2f}<br><br>
                <strong style="color:#fbbf24;">With $500 equity:</strong> Expect 1-5 shares per trade on stocks $20-$200.
                Focus on <strong>lower-priced, liquid stocks</strong> (AMD, SOFI, PLTR, F, BAC) for better position sizes.
            </div>
        </div>""", unsafe_allow_html=True)

    step5 = st.checkbox("**Step 5: Submit Paper Order** — Practice without risking real money", key="mr_step5")
    if step5:
        st.info("Go to the **Robinhood** tab. PAPER_MODE is ON — orders are simulated. The trade will be logged in your Journal.")

    step6 = st.checkbox("**Step 6: Set Exit Plan** — Know your stop and target BEFORE entering", key="mr_step6")
    if step6:
        st.markdown(f"""<div class="glass-card">
            <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7;">
                <strong style="color:#34d399;">Exit Rules:</strong><br>
                • <strong>Hard stop:</strong> Entry − {stop_atr_mult}× ATR. Non-negotiable.<br>
                • <strong>1R target (partial):</strong> Entry + 1× risk. Sell half here.<br>
                • <strong>2R target (full):</strong> Entry + 2× risk. Close rest here.<br>
                • <strong>Trailing stop:</strong> After 1R, trail at current price − {stop_atr_mult}× ATR.<br>
                • <strong>Time stop:</strong> If no movement in 5 days, re-evaluate.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Pre-Trade Checklist")
    checks = [
        "Market regime is Uptrend or Strong Uptrend",
        "Stock scores A or B grade in screener",
        "Signal scanner confirms entry today",
        "Position size is within risk budget",
        "I know my stop price BEFORE entering",
        "I know my profit target BEFORE entering",
        "I am NOT revenge trading or chasing",
        "I can afford to lose this entire risk amount",
    ]
    all_checked = True
    for i, check in enumerate(checks):
        if not st.checkbox(check, key=f"pretrade_{i}"):
            all_checked = False

    if all_checked:
        st.success("✅ All checks passed. You are ready to trade.")
    else:
        st.warning("Complete all checklist items before entering a trade.")

    st.markdown("---")
    st.markdown("#### 💡 Best Stocks to Trade with $500")
    st.markdown("""<div class="glass-card">
        <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.8;">
            <strong style="color:#22d3ee;">Best approach for small accounts:</strong><br><br>
            <strong>1. Focus on liquid stocks $10-$50:</strong> AMD, SOFI, PLTR, BAC, F, INTC, SNAP, NIO<br>
            &nbsp;&nbsp;&nbsp;→ You can buy 5-20 shares and still have meaningful position sizes.<br><br>
            <strong>2. Avoid high-priced stocks >$300:</strong> With $500, buying 1 share of a high-priced stock can over-concentrate your account and reduce flexibility.<br><br>
            <strong>3. Use the screener on cheaper stocks:</strong> Replace the default universe with:<br>
            &nbsp;&nbsp;&nbsp;<code>AMD,SOFI,PLTR,BAC,F,INTC,SNAP,HOOD,COIN,SQ,UBER,RIVN</code><br><br>
            <strong>4. Max 2 positions at once:</strong> Don't spread $500 across 5 stocks.<br><br>
            <strong>5. Paper trade for 2 weeks first:</strong> Prove the system works before going live.<br><br>
            <strong style="color:#fbbf24;">Options with $500:</strong> Only buy long calls/puts on cheap stocks. Premium ≤ $1.00 (= $100 per contract). Max 1 contract = max $100 risk.
        </div>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — STOCK SCREENER (NEW)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Stock Screener & Ranking")
    st.caption("Multi-factor scoring: Momentum · Trend · Relative Strength · Volatility · Volume")

    with st.expander("How the scoring works", expanded=False):
        st.markdown("""
**Composite Score (0–100)** weighted by:
| Factor | Weight | What it measures |
|---|---|---|
| **Momentum** | 30% | 21/63/126-day price returns |
| **Trend Quality** | 25% | Price vs SMAs, golden cross |
| **Relative Strength** | 20% | Performance vs SPY 63-day |
| **Volatility Quality** | 15% | ATR% vs historical median (lower = better entry) |
| **Volume Trend** | 10% | 20-day vs 50-day volume ratio with price confirmation |

**Grades:** A (80+) · B (65–79) · C (50–64) · D (35–49) · F (<35)

**Best approach:** Focus on A and B grades in a Bullish regime. Avoid F grades. In corrections, only trade A grades with reduced size.
        """)

    if st.button("Run Screener", type="primary", key="run_screener"):
        with st.spinner("Fetching data and scoring universe..."):
            prices = fetch_many(symbols, years=3)
            scores = rank_universe(prices, spy_df, sma_fast, sma_slow, atr_window)

        if not scores:
            st.warning("No stocks could be scored. Check symbols or data availability.")
        else:
            # Top picks row
            top_picks = [s for s in scores if s.grade in ("A", "B")]
            avoid_list = [s for s in scores if s.grade in ("D", "F")]

            if top_picks:
                st.markdown("#### 🏆 Top Picks")
                cols = st.columns(min(len(top_picks), 4))
                for i, s in enumerate(top_picks[:4]):
                    with cols[i]:
                        vc = verdict_class(s.verdict)
                        gc = grade_color(s.grade)
                        st.markdown(f"""<div class="glass-card-glow">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="font-size:1.2rem; font-weight:700; color:#f1f5f9;">{s.symbol}</span>
                                <span class="grade-badge grade-{s.grade}">{s.grade}</span>
                            </div>
                            <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:#f1f5f9; margin-bottom:4px;">${s.close:,.2f}</div>
                            <span class="verdict-pill {vc}">{s.verdict}</span>
                            <div style="margin-top:10px; font-size:0.8rem; color:#a1b0c8;">
                                Score: <span style="color:{gc}; font-weight:600;">{s.composite:.0f}</span> · RSI: {s.rsi:.0f} · {s.sma_trend}
                            </div>
                            {score_bar_html("Momentum", s.momentum, gc)}
                            {score_bar_html("Trend", s.trend_quality, gc)}
                            {score_bar_html("Rel Str", s.relative_strength, gc)}
                            {score_bar_html("Vol Qual", s.volatility_quality, gc)}
                            {score_bar_html("Volume", s.volume_trend, gc)}
                        </div>""", unsafe_allow_html=True)

            # Full ranking table
            st.markdown("#### Full Universe Ranking")
            table_rows = []
            for s in scores:
                table_rows.append({
                    "Grade": s.grade,
                    "Symbol": s.symbol,
                    "Verdict": s.verdict,
                    "Score": s.composite,
                    "Momentum": s.momentum,
                    "Trend": s.trend_quality,
                    "Rel Str": s.relative_strength,
                    "Vol Qual": s.volatility_quality,
                    "Volume": s.volume_trend,
                    "Price": s.close,
                    "ATR": s.atr,
                    "RSI": s.rsi,
                    "Trend Dir": s.sma_trend,
                })

            table_df = pd.DataFrame(table_rows)
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                    "Momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                    "Trend": st.column_config.ProgressColumn("Trend", min_value=0, max_value=100, format="%.0f"),
                    "Rel Str": st.column_config.ProgressColumn("Rel Str", min_value=0, max_value=100, format="%.0f"),
                    "Vol Qual": st.column_config.ProgressColumn("Vol Qual", min_value=0, max_value=100, format="%.0f"),
                    "Volume": st.column_config.ProgressColumn("Volume", min_value=0, max_value=100, format="%.0f"),
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "ATR": st.column_config.NumberColumn(format="$%.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                },
            )

            # Radar chart for top stock
            if scores:
                best = scores[0]
                categories = ["Momentum", "Trend Quality", "Relative Strength", "Volatility Quality", "Volume Trend"]
                values = [best.momentum, best.trend_quality, best.relative_strength, best.volatility_quality, best.volume_trend]
                values.append(values[0])
                categories.append(categories[0])

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values, theta=categories, fill="toself",
                    fillcolor="rgba(59,130,246,0.15)",
                    line=dict(color="#3b82f6", width=2),
                    name=best.symbol,
                ))
                fig = dark_plotly_layout(fig, f"{best.symbol} Factor Profile")
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(10,14,23,0.95)",
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(99,179,237,0.1)"),
                        angularaxis=dict(gridcolor="rgba(99,179,237,0.1)"),
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            if avoid_list:
                st.markdown("#### ⚠️ Avoid List")
                avoid_syms = ", ".join(f"**{s.symbol}** ({s.grade})" for s in avoid_list)
                st.warning(f"Weak scores: {avoid_syms}. Either downtrending, losing relative strength, or high volatility risk.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIGNAL SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Signal Scanner")
    if available_slots == 0:
        st.warning("Max open-position cap reached.")
    if daily_lockout:
        st.warning("Daily loss lockout active. New buys blocked.")

    with st.expander("Strategy logic", expanded=False):
        if strategy_name == "breakout":
            st.markdown("**Trend Breakout:** Close > slow SMA, fast > slow SMA, close breaks rolling high, outperforms SPY 63d.")
        elif strategy_name == "pullback":
            st.markdown("**Trend Pullback:** Close > slow SMA, fast > slow SMA, price pulling back to fast SMA, RSI > 40, outperforms SPY.")
        else:
            st.markdown("**Mean Reversion:** Close > slow SMA, price < fast SMA, RSI oversold, outperforms SPY 63d.")

    if st.button("Scan Now", type="primary", key="scan_now"):
        with st.spinner("Scanning..."):
            prices = fetch_many(symbols, years=3)
            rows = scan_universe(prices, spy_df, strategy_cfg)

        missing = [sym for sym in symbols if sym not in prices]
        if missing:
            st.warning(f"Skipped: {', '.join(missing)}")
        if not regime:
            st.warning("SPY below 200 SMA. Regime filter OFF — no new longs.")
        if not rows:
            st.info(f"No {STRATEGY_LABELS[strategy_name].lower()} signals today.")
        else:
            out = []
            for row in rows:
                if np.isnan(row["atr"]):
                    continue
                stop = row["close"] - stop_atr_mult * row["atr"]
                qty = position_size(equity, risk_per_trade, row["close"], stop)
                est_risk = max(row["close"] - stop, 0.0) * qty
                out.append({
                    "Symbol": row["symbol"],
                    "Close": round(row["close"], 2),
                    "ATR": round(row["atr"], 2),
                    "RSI": round(float(row.get("rsi", np.nan)), 2) if not np.isnan(float(row.get("rsi", np.nan))) else np.nan,
                    "Stop": round(stop, 2),
                    "Qty": qty,
                    "Est Risk $": round(est_risk, 2),
                    "Position $": round(row["close"] * qty, 2),
                })

            if not out:
                st.info("Signals found but ATR data incomplete.")
            else:
                scan_df = pd.DataFrame(out).sort_values(["Est Risk $", "Position $"], ascending=[True, True])
                spy_ind = add_indicators(spy_df, strategy_cfg)
                spy_ret_63 = float(spy_ind["RET_63"].iloc[-1]) if not spy_ind.empty else 0.0

                ranked_rows = []
                for sym in scan_df["Symbol"].tolist():
                    sym_df = prices.get(sym)
                    if sym_df is None or sym_df.empty:
                        continue
                    ind = add_indicators(sym_df, strategy_cfg)
                    last = ind.iloc[-1]
                    rel_63 = float((last.get("RET_63", 0.0) - spy_ret_63) * 100.0)
                    atr_pct = float((last["ATR"] / last["Close"]) * 100.0) if last["Close"] > 0 else 0.0

                    if strategy_name == "breakout":
                        setup_metric = float((last["Close"] / last["HH_BREAKOUT"] - 1.0) * 100.0) if not pd.isna(last.get("HH_BREAKOUT")) and last["HH_BREAKOUT"] > 0 else 0.0
                        setup_label = "Breakout %"
                    elif strategy_name == "pullback":
                        setup_metric = float(((last["SMA_FAST"] - last["Close"]) / last["Close"]) * 100.0)
                        setup_label = "Pullback %"
                    else:
                        rsi_now = float(last.get("RSI", 50.0)) if not pd.isna(last.get("RSI")) else 50.0
                        setup_metric = max(0.0, 35.0 - rsi_now)
                        setup_label = "Oversold Score"

                    score = (0.55 * rel_63) + (0.35 * setup_metric) - (0.10 * atr_pct)
                    ranked_rows.append({
                        "Symbol": sym,
                        "Score": round(score, 2),
                        "Rel 63d vs SPY %": round(rel_63, 2),
                        setup_label: round(setup_metric, 2),
                        "ATR %": round(atr_pct, 2),
                    })

                if ranked_rows:
                    top3 = pd.DataFrame(ranked_rows).sort_values("Score", ascending=False).head(3)
                    st.markdown("#### 🏅 Top 3 Trades Today")
                    st.dataframe(top3, use_container_width=True, hide_index=True, column_config={
                        "Score": st.column_config.NumberColumn(format="%.2f"),
                        "Rel 63d vs SPY %": st.column_config.NumberColumn(format="%.2f%%"),
                        "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
                    })

                st.dataframe(scan_df, use_container_width=True, column_config={
                    "Close": st.column_config.NumberColumn(format="$%.2f"),
                    "ATR": st.column_config.NumberColumn(format="$%.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.2f"),
                    "Stop": st.column_config.NumberColumn(format="$%.2f"),
                    "Est Risk $": st.column_config.NumberColumn(format="$%.2f"),
                    "Position $": st.column_config.NumberColumn(format="$%.2f"),
                })

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Backtest Engine")
    st.caption("Uses the same strategy settings from sidebar.")
    bt_symbol = st.text_input("Ticker", value="AAPL").upper().strip()
    fee_bps = st.slider("Slippage + Fees (bps)", 0, 30, 5)
    trailing_mode = st.radio("Trailing Stop", options=["On", "Off", "Compare Both"], horizontal=True, index=2)

    if st.button("Run Backtest", key="run_backtest"):
        bt_df = fetch_prices(bt_symbol, years=5)
        if bt_df.empty or len(bt_df) < 300:
            st.error("Not enough data.")
        else:
            base_cfg = dict(
                fee_bps=fee_bps, strategy_name=strategy_name, breakout_window=breakout_window,
                sma_fast=sma_fast, sma_slow=sma_slow, atr_window=atr_window, stop_atr_mult=stop_atr_mult,
            )
            curve_trail, stats_trail = run_backtest(bt_df, BacktestConfig(use_trailing_stop=True, **base_cfg))
            curve_static, stats_static = run_backtest(bt_df, BacktestConfig(use_trailing_stop=False, **base_cfg))

            def calc_metrics(curve_df, stats_dict):
                vol = float(curve_df["strategy_ret"].std())
                sharpe = float(np.sqrt(252.0) * curve_df["strategy_ret"].mean() / vol) if vol > 0 else 0.0
                years = max(len(curve_df) / 252.0, 1e-9)
                cagr = float((curve_df["equity_curve"].iloc[-1] ** (1.0 / years)) - 1.0)
                return {
                    "Strategy Return": float(stats_dict["Strategy Return"]),
                    "Buy/Hold Return": float(stats_dict["Buy/Hold Return"]),
                    "Max Drawdown": float(stats_dict["Max Drawdown"]),
                    "CAGR": cagr, "Sharpe": sharpe,
                }

            mt = calc_metrics(curve_trail, stats_trail)
            ms = calc_metrics(curve_static, stats_static)

            max_drawdown_cap = -0.25
            candidates = []
            if mt["Max Drawdown"] >= max_drawdown_cap:
                candidates.append(("Trailing Stop ON", mt))
            if ms["Max Drawdown"] >= max_drawdown_cap:
                candidates.append(("Trailing Stop OFF", ms))

            if candidates:
                best_name, best_metrics = max(candidates, key=lambda x: x[1]["Sharpe"])
                st.markdown(
                    f"<div class='recommend'><strong>Recommended:</strong> {best_name} — Sharpe {best_metrics['Sharpe']:.2f}, Drawdown {best_metrics['Max Drawdown']*100:.2f}%</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"No mode met drawdown cap ({max_drawdown_cap*100:.0f}%).")

            if trailing_mode == "On":
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Return", f"{mt['Strategy Return']*100:.2f}%")
                m2.metric("Buy/Hold", f"{mt['Buy/Hold Return']*100:.2f}%")
                m3.metric("Drawdown", f"{mt['Max Drawdown']*100:.2f}%")
                m4.metric("CAGR", f"{mt['CAGR']*100:.2f}%")
                m5.metric("Sharpe", f"{mt['Sharpe']:.2f}")
            elif trailing_mode == "Off":
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Return", f"{ms['Strategy Return']*100:.2f}%")
                m2.metric("Buy/Hold", f"{ms['Buy/Hold Return']*100:.2f}%")
                m3.metric("Drawdown", f"{ms['Max Drawdown']*100:.2f}%")
                m4.metric("CAGR", f"{ms['CAGR']*100:.2f}%")
                m5.metric("Sharpe", f"{ms['Sharpe']:.2f}")
            else:
                cl, cr = st.columns(2)
                with cl:
                    st.markdown("**Trailing ON**")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Ret", f"{mt['Strategy Return']*100:.2f}%")
                    m2.metric("B/H", f"{mt['Buy/Hold Return']*100:.2f}%")
                    m3.metric("DD", f"{mt['Max Drawdown']*100:.2f}%")
                    m4.metric("CAGR", f"{mt['CAGR']*100:.2f}%")
                    m5.metric("Sharpe", f"{mt['Sharpe']:.2f}")
                with cr:
                    st.markdown("**Trailing OFF**")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Ret", f"{ms['Strategy Return']*100:.2f}%")
                    m2.metric("B/H", f"{ms['Buy/Hold Return']*100:.2f}%")
                    m3.metric("DD", f"{ms['Max Drawdown']*100:.2f}%")
                    m4.metric("CAGR", f"{ms['CAGR']*100:.2f}%")
                    m5.metric("Sharpe", f"{ms['Sharpe']:.2f}")
                delta_ret = mt["Strategy Return"] - ms["Strategy Return"]
                delta_dd = mt["Max Drawdown"] - ms["Max Drawdown"]
                st.info(f"Delta (ON − OFF): Return {delta_ret*100:+.2f}%, Drawdown {delta_dd*100:+.2f}%")

            chart_df = pd.DataFrame({
                "Date": curve_trail.index,
                "buy_hold": curve_trail["buy_hold"],
                "trailing_on": curve_trail["equity_curve"],
                "trailing_off": curve_static["equity_curve"],
            })
            if trailing_mode == "On":
                y_cols = ["trailing_on", "buy_hold"]
                ttl = f"{bt_symbol} (Trailing ON)"
            elif trailing_mode == "Off":
                y_cols = ["trailing_off", "buy_hold"]
                ttl = f"{bt_symbol} (Trailing OFF)"
            else:
                y_cols = ["trailing_on", "trailing_off", "buy_hold"]
                ttl = f"{bt_symbol} Comparison"

            fig = px.line(chart_df, x="Date", y=y_cols, title=ttl)
            fig = dark_plotly_layout(fig, ttl)
            fig.update_traces(line=dict(width=2.4))
            st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROBINHOOD
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Robinhood Execution")
    st.caption("Keep PAPER_MODE=true until fully validated.")

    env_user, env_pass, env_totp_secret = env_credentials()
    rh_user = st.text_input("Username", value=env_user or "")
    rh_pass = st.text_input("Password", value=env_pass or "", type="password")
    mfa_code = st.text_input("MFA Code", help="Current verification code if prompted.")
    totp_secret = st.text_input("TOTP Secret", value=env_totp_secret or "", type="password")

    st.markdown(
        f"<div class='status-row'><strong>Routing:</strong> {'Simulation (PAPER)' if paper_mode else 'LIVE orders'} · {len(open_df)}/{int(max_open_positions)} open · Realized ${today_realized_pnl:,.2f}</div>",
        unsafe_allow_html=True,
    )

    if "rh_client" not in st.session_state:
        st.session_state["rh_client"] = RobinhoodClient()

    if st.button("Login", key="login_rh"):
        try:
            ok = st.session_state["rh_client"].login(rh_user, rh_pass, mfa_code=mfa_code or None, totp_secret=totp_secret or None)
            if ok:
                bp = st.session_state["rh_client"].profile_buying_power()
                st.success(f"Logged in. Buying power: ${bp:,.2f}" if bp else "Logged in.")
            else:
                st.error("Login failed.")
        except Exception as exc:
            st.error(f"Login error: {exc}")

    st.markdown("---")
    st.markdown("#### Order Ticket")
    o1, o2, o3 = st.columns(3)
    order_symbol = o1.text_input("Symbol", value="AAPL", key="order_sym").upper().strip()
    side = o2.selectbox("Side", ["buy", "sell"])
    qty = o3.number_input("Quantity", min_value=1, value=1, step=1)
    limit_price = st.number_input("Limit Price (0=market)", min_value=0.0, value=0.0, step=0.01)

    if st.button("Submit Order", type="primary", key="submit_order"):
        try:
            if side == "buy":
                if daily_lockout:
                    raise ValueError("Blocked by daily loss lockout.")
                if len(open_df) >= int(max_open_positions):
                    raise ValueError(f"Max open positions ({int(max_open_positions)}) reached.")
                if not regime:
                    raise ValueError("SPY below 200 SMA — regime filter blocks buys.")

                px_df = fetch_prices(order_symbol, years=2)
                if px_df.empty or len(px_df) < max(atr_window + 5, 60):
                    raise ValueError(f"Not enough data for {order_symbol}.")
                ind_df = add_indicators(px_df, strategy_cfg)
                latest = ind_df.iloc[-1]
                latest_close = float(latest["Close"])
                latest_atr = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else np.nan
                if np.isnan(latest_atr) or latest_atr <= 0:
                    raise ValueError(f"ATR unavailable for {order_symbol}.")

                entry_price = float(limit_price) if limit_price > 0 else latest_close
                stop_price = entry_price - (stop_atr_mult * latest_atr)
                if stop_price <= 0:
                    raise ValueError("Invalid stop price.")
                risk_per_share = max(entry_price - stop_price, 0.0)
                target_price = entry_price + (2.0 * risk_per_share)
                partial_target = entry_price + risk_per_share
                estimated_risk = risk_per_share * int(qty)
                risk_budget = equity * risk_per_trade
                max_qty = position_size(equity, risk_per_trade, entry_price, stop_price)

                if int(qty) > max_qty:
                    raise ValueError(f"Max qty = {max_qty} at this risk budget.")
                if estimated_risk > (risk_budget + 1e-9):
                    raise ValueError(f"Risk ${estimated_risk:,.2f} > budget ${risk_budget:,.2f}.")

                st.info(f"Entry ${entry_price:,.2f} · Stop ${stop_price:,.2f} · Target ${target_price:,.2f} · Risk ${estimated_risk:,.2f}")

            req = RobinhoodOrderRequest(symbol=order_symbol, side=side, quantity=int(qty), limit_price=float(limit_price) if limit_price > 0 else None)
            result = st.session_state["rh_client"].submit_order(req, paper_mode=paper_mode)
            if paper_mode:
                st.warning("PAPER MODE — no live order sent.")
            if side == "buy":
                append_trade(TradeRecord(
                    symbol=order_symbol, side=side, quantity=int(qty),
                    entry_price=float(entry_price), stop_price=float(stop_price),
                    target_price=float(target_price), risk_amount=float(estimated_risk),
                    paper_mode=paper_mode,
                    strategy_tag=f"{strategy_name}_{breakout_window}_{sma_fast}_{sma_slow}",
                    notes=f"Partial 1R at {partial_target:.2f}; trail stop after.",
                ))
                st.success("Trade saved to journal.")
            st.json(result)
        except Exception as exc:
            st.error(f"Order error: {exc}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADE JOURNAL
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Trade Journal")

    j1, j2, j3, j4, j5 = st.columns(5)
    j1.metric("Open", f"{journal_summary['Open Positions']}")
    j2.metric("Closed", f"{journal_summary['Closed Trades']}")
    j3.metric("Win Rate", f"{journal_summary['Win Rate']*100:.1f}%")
    j4.metric("Net PnL", f"${journal_summary['Net PnL']:,.2f}")
    j5.metric("Expectancy", f"${journal_summary['Expectancy']:,.2f}")

    st.markdown(
        f"<div class='status-row'><strong>Summary:</strong> Avg winner ${journal_summary['Avg Winner']:,.2f} · Avg loser ${journal_summary['Avg Loser']:,.2f} · Today ${today_realized_pnl:,.2f}</div>",
        unsafe_allow_html=True,
    )

    if open_df.empty:
        st.info("No open positions. New buy orders appear here automatically.")
    else:
        position_manager_df = build_position_manager(open_df, strategy_cfg)
        if not position_manager_df.empty:
            st.markdown("#### Open Position Manager")
            st.dataframe(position_manager_df, use_container_width=True, hide_index=True, column_config={
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Current": st.column_config.NumberColumn(format="$%.2f"),
                "Stop": st.column_config.NumberColumn(format="$%.2f"),
                "Partial 1R": st.column_config.NumberColumn(format="$%.2f"),
                "Target 2R": st.column_config.NumberColumn(format="$%.2f"),
                "Trail Stop": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized PnL": st.column_config.NumberColumn(format="$%.2f"),
                "R": st.column_config.NumberColumn(format="%.2fR"),
            })

        st.markdown("#### Close Trade")
        open_choices = {
            f"{row['symbol']} | {int(row['quantity'])} sh | {row['trade_id']}": row['trade_id']
            for _, row in open_df.iterrows()
        }
        selected_trade = st.selectbox("Open trade", list(open_choices.keys()))
        selected_row = open_df[open_df["trade_id"] == open_choices[selected_trade]].iloc[0]
        exit_cols = st.columns(3)
        exit_price = exit_cols[0].number_input("Exit Price", min_value=0.0, value=round(float(selected_row["target_price"]), 2), step=0.01)
        exit_reason = exit_cols[1].selectbox("Exit Reason", ["stop", "partial_profit", "target", "trailing_stop", "manual_review"])
        exit_note = exit_cols[2].text_input("Note", value="")
        if st.button("Close Trade", key="close_trade"):
            try:
                close_position(open_choices[selected_trade], float(exit_price), exit_reason, notes=exit_note)
                st.success("Trade closed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

    closed_df = journal_df[journal_df["status"] == "CLOSED"].copy()
    if not closed_df.empty:
        closed_df["closed_at"] = pd.to_datetime(closed_df["closed_at"], errors="coerce")
        closed_df["pnl"] = pd.to_numeric(closed_df["pnl"], errors="coerce").fillna(0.0)
        closed_df = closed_df.sort_values("closed_at")
        closed_df["cumulative_pnl"] = closed_df["pnl"].cumsum()

        fig = px.line(closed_df, x="closed_at", y="cumulative_pnl", title="Cumulative PnL")
        fig = dark_plotly_layout(fig, "Cumulative PnL")
        fig.update_traces(line=dict(color="#10b981", width=2.5))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Closed Log")
        st.dataframe(
            closed_df[["trade_id", "symbol", "quantity", "entry_price", "exit_price", "pnl", "exit_reason", "closed_at"]],
            use_container_width=True, hide_index=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                "pnl": st.column_config.NumberColumn("PnL", format="$%.2f"),
            },
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — OPTIONS ADVISOR (NEW)
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Options Strategy Advisor")
    st.caption("Smart strategy selection based on your market outlook and volatility environment.")

    # Golden Rules
    with st.expander("📜 10 Golden Rules for Options Success", expanded=False):
        for i, rule in enumerate(OPTIONS_RULES, 1):
            st.markdown(f'<div class="rule-item"><strong>{i}.</strong> {rule}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Strategy Decision Engine
    st.markdown("#### Strategy Decision Engine")
    ac1, ac2 = st.columns(2)
    opt_symbol = ac1.text_input("Symbol to Analyze", value="SPY", key="adv_symbol").upper().strip()
    outlook = ac2.selectbox("Your Market Outlook", ["bullish", "bearish", "neutral", "volatile"], key="adv_outlook")

    if st.button("Analyze & Recommend", type="primary", key="analyze_options"):
        with st.spinner("Analyzing..."):
            opt_df = fetch_prices(opt_symbol, years=3)

        if opt_df.empty or len(opt_df) < 60:
            st.error(f"Not enough data for {opt_symbol}.")
        else:
            result = options_decision_summary(opt_symbol, opt_df, outlook)

            # IV Rank Gauge
            iv = result["iv_rank"]
            iv_color = "#10b981" if iv < 30 else ("#f59e0b" if iv < 60 else "#ef4444")
            iv_label = "LOW — Buy premium" if iv < 30 else ("MODERATE" if iv < 60 else "HIGH — Sell premium")

            st.markdown(f"""<div class="glass-card-glow">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.4rem; font-weight:700; color:#f1f5f9;">{result['symbol']}</span>
                        <span style="margin-left:12px; font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#f1f5f9;">${result['close']:,.2f}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:#a1b0c8; text-transform:uppercase; letter-spacing:0.5px;">5d / 21d Return</div>
                        <span style="font-family:'JetBrains Mono',monospace; color:{'#10b981' if result['ret_5d'] >= 0 else '#ef4444'};">{result['ret_5d']:+.2f}%</span>
                        <span style="margin-left:8px; font-family:'JetBrains Mono',monospace; color:{'#10b981' if result['ret_21d'] >= 0 else '#ef4444'};">{result['ret_21d']:+.2f}%</span>
                    </div>
                </div>
                <div style="margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:0.8rem; color:#a1b0c8;">IV Rank (HV Proxy)</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:{iv_color};">{iv:.0f} — {iv_label}</span>
                    </div>
                    <div class="iv-bar-container">
                        <div class="iv-bar-fill" style="width:{iv}%; background:linear-gradient(90deg, #10b981, #f59e0b, #ef4444);"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#a1b0c8; margin-top:2px;">
                        <span>HV Low: {result['hv_1y_low']:.1f}%</span>
                        <span>Current: {result['current_hv']:.1f}%</span>
                        <span>HV High: {result['hv_1y_high']:.1f}%</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Recommended Strategies
            st.markdown("#### Recommended Strategies")
            st.caption(f"Based on **{outlook.upper()}** outlook with IV rank at **{iv:.0f}**")

            for rec in result["recommendations"]:
                card_class = "strat-card-best" if rec.priority == 1 else "strat-card"
                badge = '<span style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); padding:2px 8px; border-radius:999px; font-size:0.7rem; font-weight:600; margin-left:8px;">BEST FIT</span>' if rec.priority == 1 else ""
                st.markdown(f"""<div class="{card_class}">
                    <div style="display:flex; align-items:center; margin-bottom:6px;">
                        <span style="font-size:1.05rem; font-weight:700; color:#22d3ee;">{rec.strategy}</span>
                        {badge}
                    </div>
                    <div style="font-size:0.88rem; color:#cbd5e1; margin-bottom:8px;">{rec.reason}</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.82rem;">
                        <div><span style="color:#a1b0c8;">Risk:</span> <span style="color:#f1f5f9;">{rec.risk_profile}</span></div>
                        <div><span style="color:#a1b0c8;">Best IV:</span> <span style="color:#f1f5f9;">{rec.ideal_iv}</span></div>
                        <div><span style="color:#a1b0c8;">Probability:</span> <span style="color:#f1f5f9;">{rec.probability_description}</span></div>
                        <div><span style="color:#a1b0c8;">Example:</span> <span style="color:#fbbf24;">{rec.example}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Decision Flowchart
            st.markdown("#### Decision Flowchart")
            st.markdown(f"""<div class="glass-card">
                <pre style="color:#cbd5e1; font-family:'JetBrains Mono',monospace; font-size:0.8rem; line-height:1.6; margin:0;">
┌─────────────────────────────────────────────────┐
│              YOUR OUTLOOK: {outlook.upper():<10}            │
│              IV RANK:      {iv:>5.0f}               │
├─────────────────────────────────────────────────┤
│                                                 │
│  IV LOW (&lt;30)          │  IV HIGH (&gt;50)         │
│  ═══════════           │  ════════════          │
│  BUY premium           │  SELL premium          │
│  • Long calls/puts     │  • Credit spreads      │
│  • Debit spreads       │  • Iron condors        │
│  • Straddles           │  • Covered strategies  │
│                        │                        │
│  WHY: Options cheap,   │  WHY: Options rich,    │
│  profit from IV rise   │  profit from IV crush  │
│  + directional move    │  + time decay          │
│                                                 │
├─────────────────────────────────────────────────┤
│  SIZING RULE: Max loss per trade ≤ 1-2% equity  │
│  TIMING: 30-45 DTE optimal for theta balance    │
│  EXIT: Close winners at 50% max profit          │
└─────────────────────────────────────────────────┘
                </pre>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Options Calculator (existing functionality preserved)
    st.markdown("#### Options Risk Calculator")
    risk_budget = equity * risk_per_trade
    options_symbol = st.text_input("Underlying", value="SPY", key="opt_symbol").upper().strip()
    underlying_price = st.number_input("Underlying Price", min_value=0.01, value=500.0, step=0.5, key="opt_underlying")
    options_strategy = st.selectbox("Strategy", [
        "Long Call", "Long Put", "Bull Call Debit Spread", "Bear Put Debit Spread",
        "Bull Put Credit Spread", "Bear Call Credit Spread",
    ], key="opt_strategy")
    contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1, step=1, key="opt_contracts")

    max_loss = max_profit = breakeven = 0.0
    max_loss_per_contract = 0.0
    plan_valid = True
    plan_notes: list[str] = []

    if options_strategy in {"Long Call", "Long Put"}:
        strike = st.number_input("Strike", min_value=0.01, value=500.0, step=0.5, key="opt_strike")
        premium = st.number_input("Premium (per share)", min_value=0.01, value=5.0, step=0.05, key="opt_premium")
        if options_strategy == "Long Call":
            max_loss, max_profit, breakeven, max_loss_per_contract = calc_long_call(strike, premium, int(contracts))
        else:
            max_loss, max_profit, breakeven, max_loss_per_contract = calc_long_put(strike, premium, int(contracts))
    elif options_strategy == "Bull Call Debit Spread":
        c1, c2, c3 = st.columns(3)
        ls = c1.number_input("Long Strike", value=495.0, step=0.5, key="bcd_l")
        ss = c2.number_input("Short Strike", value=505.0, step=0.5, key="bcd_s")
        db = c3.number_input("Net Debit", value=3.0, step=0.05, key="bcd_d")
        if ss <= ls:
            plan_valid = False
            plan_notes.append("Short > Long strike required.")
        max_loss, max_profit, breakeven, max_loss_per_contract = calc_bull_call_debit(ls, ss, db, int(contracts))
    elif options_strategy == "Bear Put Debit Spread":
        c1, c2, c3 = st.columns(3)
        ls = c1.number_input("Long Strike", value=505.0, step=0.5, key="bpd_l")
        ss = c2.number_input("Short Strike", value=495.0, step=0.5, key="bpd_s")
        db = c3.number_input("Net Debit", value=3.0, step=0.05, key="bpd_d")
        if ls <= ss:
            plan_valid = False
            plan_notes.append("Long > Short strike required.")
        max_loss, max_profit, breakeven, max_loss_per_contract = calc_bear_put_debit(ls, ss, db, int(contracts))
    elif options_strategy == "Bull Put Credit Spread":
        c1, c2, c3 = st.columns(3)
        sp = c1.number_input("Short Put", value=500.0, step=0.5, key="bpc_s")
        lp = c2.number_input("Long Put", value=495.0, step=0.5, key="bpc_l")
        cr = c3.number_input("Net Credit", value=1.5, step=0.05, key="bpc_c")
        if sp <= lp:
            plan_valid = False
            plan_notes.append("Short put > Long put required.")
        max_loss, max_profit, breakeven, max_loss_per_contract = calc_bull_put_credit(sp, lp, cr, int(contracts))
    else:
        c1, c2, c3 = st.columns(3)
        sc = c1.number_input("Short Call", value=500.0, step=0.5, key="bcc_s")
        lc = c2.number_input("Long Call", value=505.0, step=0.5, key="bcc_l")
        cr = c3.number_input("Net Credit", value=1.5, step=0.05, key="bcc_c")
        if lc <= sc:
            plan_valid = False
            plan_notes.append("Long call > Short call required.")
        max_loss, max_profit, breakeven, max_loss_per_contract = calc_bear_call_credit(sc, lc, cr, int(contracts))

    suggested_contracts = max_contracts_for_risk(max_loss_per_contract, risk_budget)
    rr = None
    if max_profit is not None and max_profit and max_loss > 0:
        rr = max_profit / max_loss

    if max_loss > risk_budget + 1e-9:
        plan_valid = False
        plan_notes.append(f"Max loss ${max_loss:,.2f} > budget ${risk_budget:,.2f}.")

    q1 = st.checkbox("Liquidity check (tight bid/ask)", value=True, key="chk1")
    q2 = st.checkbox("Earnings/event risk reviewed", value=True, key="chk2")
    q3 = st.checkbox("Exit plan defined", value=True, key="chk3")
    q4 = st.checkbox("Portfolio risk within limits", value=True, key="chk4")
    if not (q1 and q2 and q3 and q4):
        plan_valid = False
        plan_notes.append("Checklist incomplete.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Budget", f"${risk_budget:,.2f}")
    m2.metric("Max Loss", f"${max_loss:,.2f}")
    m3.metric("Max Profit", "Unbounded" if max_profit is None else f"${max_profit:,.2f}")
    m4.metric("Breakeven", "N/A" if breakeven is None else f"${breakeven:,.2f}")
    m5.metric("Suggested Qty", f"{suggested_contracts}")
    if rr is not None:
        st.info(f"Risk/Reward: {rr:.2f}x")
    if plan_valid:
        st.success("Plan passed all gates.")
    else:
        for msg in plan_notes:
            st.error(msg)

    if st.button("Save Plan", key="save_plan"):
        plan = OptionPlan(
            symbol=options_symbol, strategy=options_strategy, contracts=int(contracts),
            underlying_price=float(underlying_price), max_loss=float(max_loss),
            max_profit=None if max_profit is None else float(max_profit),
            breakeven=None if breakeven is None else float(breakeven),
            risk_budget=float(risk_budget), suggested_contracts=int(suggested_contracts),
            valid=bool(plan_valid), notes=" | ".join(plan_notes) if plan_notes else "Validated",
        )
        append_options_plan(plan)
        st.success("Plan saved.")

    plans_df = load_options_plans()
    if not plans_df.empty:
        st.markdown("#### Plan History")
        st.dataframe(plans_df.tail(30).iloc[::-1], use_container_width=True, hide_index=True, column_config={
            "underlying_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "max_loss": st.column_config.NumberColumn("Max Loss", format="$%.2f"),
            "risk_budget": st.column_config.NumberColumn("Budget", format="$%.2f"),
        })

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — TRADINGVIEW ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 📺 TradingView Technical Analysis")
    st.caption("Live technical signals from TradingView for your universe stocks.")

    # ── TradingView Technical Data (scraped Apr 14, 2026) ──
    _tv_data = {
        "SPY":  {"price": 686.10, "chg": "+0.98%", "rsi": 63.33, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S2/N5/B4)", "ma_summary": "Buy (S1/N1/B13)", "overall": "Buy (S3/N6/B17)", "strategy": "Trend Breakout", "note": "SPY above all key MAs. RSI 63 = healthy uptrend, not overbought. Strong MA alignment = bullish."},
        "NVDA": {"price": 189.31, "chg": "+0.36%", "rsi": 61.99, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S1/N7/B3)", "ma_summary": "Buy (S0/N1/B14)", "overall": "Buy (S1/N8/B17)", "strategy": "Trend Breakout", "note": "All 15 MAs say Buy. RSI 62 = strong momentum. Above EMA200 ($174). Best breakout candidate."},
        "AMD":  {"price": 246.83, "chg": "+0.73%", "rsi": 70.82, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S1/N7/B3)", "ma_summary": "Buy (S1/N1/B13)", "overall": "Buy (S2/N8/B16)", "strategy": "Trend Breakout", "note": "RSI 71 = entering overbought. MACD strong positive. Wait for RSI pullback to 60 or breakout above $250 for entry."},
        "INTC": {"price": 65.18, "chg": "+4.49%", "rsi": 77.84, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S2/N6/B3)", "ma_summary": "Buy (S1/N1/B13)", "overall": "Buy (S3/N7/B16)", "strategy": "Mean Reversion CAUTION", "note": "RSI 78 = OVERBOUGHT. Massive +4.5% day. All MAs bullish but extended. Wait for pullback to EMA20 ($52) before entry."},
        "PLTR": {"price": 132.37, "chg": "+3.37%", "rsi": 38.85, "macd": "Sell", "momentum": "Buy", "osc_summary": "Neutral (S1/N8/B2)", "ma_summary": "Sell (S13/N1/B1)", "overall": "Sell (S14/N9/B3)", "strategy": "AVOID", "note": "13 of 15 MAs say Sell. RSI 39 = weak. Below all major MAs. MACD negative. Wait for trend reversal."},
        "SOFI": {"price": 17.05, "chg": "+5.12%", "rsi": 49.57, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S0/N8/B3)", "ma_summary": "Neutral (S8/N1/B6)", "overall": "Neutral (S8/N9/B9)", "strategy": "Pullback Watch", "note": "Below EMA50 ($18.64) and EMA200 ($21.26). RSI 50 = neutral. Big +5% bounce but still in downtrend. Wait for close above $18.50."},
        "META": {"price": 634.53, "chg": "+0.74%", "rsi": 58.65, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S0/N9/B2)", "ma_summary": "Neutral (S5/N1/B9)", "overall": "Neutral (S5/N10/B11)", "strategy": "Pullback Buy", "note": "Below EMA100/200 but above shorter MAs. RSI 59 = moderate. Could bounce off EMA30 support ($609). Watch $640 resistance."},
        "AAPL": {"price": 259.20, "chg": "-0.49%", "rsi": 53.64, "macd": "Buy", "momentum": "Buy", "osc_summary": "Neutral (S2/N7/B2)", "ma_summary": "Neutral (S3/N1/B11)", "overall": "Neutral (S5/N8/B13)", "strategy": "Pullback Buy", "note": "RSI 54 = neutral zone. Above EMA200 ($252). SMA50 ($261) acting as resistance. Buy on dip to $252-$255."},
    }

    # ── Summary Cards ──
    st.markdown("#### Market Overview (Apr 14, 2026 — Close)")

    # Rank stocks by signal strength
    _signal_order = {"Buy": 3, "Neutral": 2, "Sell": 1, "AVOID": 0}
    _sorted_tv = sorted(_tv_data.items(), key=lambda x: _signal_order.get(x[1]["overall"].split(" ")[0], 0), reverse=True)

    # Top picks
    st.markdown("""<div class="glass-card-glow">
        <span style="font-size:1.1rem;font-weight:700;color:#34d399;">🏆 TOP PICKS FROM TRADINGVIEW</span><br>
        <span style="color:#f1f5f9;font-weight:600;">1. NVDA $189.31</span> — <span style="color:#34d399;">Strong Buy</span> — All 14/15 MAs bullish, RSI 62, best breakout setup<br>
        <span style="color:#f1f5f9;font-weight:600;">2. AMD $246.83</span> — <span style="color:#34d399;">Buy</span> — Strong momentum, RSI slightly hot at 71, wait for $250 breakout<br>
        <span style="color:#f1f5f9;font-weight:600;">3. AAPL $259.20</span> — <span style="color:#60a5fa;">Neutral/Buy</span> — Safe pullback entry at $252-255 support zone<br>
        <br>
        <span style="font-size:0.85rem;color:#f87171;font-weight:600;">⚠️ AVOID: PLTR $132.37</span> — <span style="color:#fca5a5;">13/15 MAs say Sell, RSI weak at 39, below all major MAs</span>
    </div>""", unsafe_allow_html=True)

    # ── Detailed Table ──
    st.markdown("#### Detailed Technical Signals")

    for sym, d in _sorted_tv:
        overall_signal = d["overall"].split(" ")[0]
        if overall_signal == "Buy":
            border_color = "rgba(52,211,153,0.4)"
            signal_color = "#34d399"
            card_class = "strat-card-best"
        elif overall_signal == "Sell" or overall_signal == "AVOID":
            border_color = "rgba(248,113,113,0.4)"
            signal_color = "#f87171"
            card_class = "strat-card"
        else:
            border_color = "rgba(96,165,250,0.4)"
            signal_color = "#60a5fa"
            card_class = "strat-card"

        rsi_color = "#f87171" if d["rsi"] > 70 else "#34d399" if d["rsi"] < 40 else "#fbbf24" if d["rsi"] > 60 else "#60a5fa"

        st.markdown(f"""<div class="{card_class}" style="border-color:{border_color};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div>
                    <span style="font-size:1.2rem;font-weight:800;color:#ffffff;">{sym}</span>
                    <span style="color:#cbd5e1;margin-left:8px;">${d["price"]}</span>
                    <span style="color:{'#34d399' if '+' in d['chg'] else '#f87171'};margin-left:6px;font-weight:600;">{d["chg"]}</span>
                </div>
                <div>
                    <span style="padding:4px 12px;border-radius:999px;font-size:0.8rem;font-weight:700;color:{signal_color};background:rgba(0,0,0,0.3);border:1px solid {border_color};">{d["overall"]}</span>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
                <div style="text-align:center;"><span style="color:#a1b0c8;font-size:0.75rem;">RSI(14)</span><br><span style="color:{rsi_color};font-weight:700;font-size:1rem;">{d["rsi"]}</span></div>
                <div style="text-align:center;"><span style="color:#a1b0c8;font-size:0.75rem;">MACD</span><br><span style="color:{'#34d399' if d['macd']=='Buy' else '#f87171'};font-weight:700;">{d["macd"]}</span></div>
                <div style="text-align:center;"><span style="color:#a1b0c8;font-size:0.75rem;">Oscillators</span><br><span style="color:#cbd5e1;font-weight:600;font-size:0.85rem;">{d["osc_summary"]}</span></div>
                <div style="text-align:center;"><span style="color:#a1b0c8;font-size:0.75rem;">Moving Avgs</span><br><span style="color:#cbd5e1;font-weight:600;font-size:0.85rem;">{d["ma_summary"]}</span></div>
            </div>
            <div style="margin-top:4px;">
                <span style="color:#22d3ee;font-weight:600;">Strategy:</span> <span style="color:#f1f5f9;">{d["strategy"]}</span><br>
                <span style="color:#a1b0c8;font-size:0.85rem;">{d["note"]}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Strategy Recommendations for $500 Account ──
    st.markdown("#### 🎯 Best Strategies for Your $500 Account")

    st.markdown("""<div class="glass-card-glow">
        <span style="font-size:1rem;font-weight:700;color:#22d3ee;">STRATEGY 1: NVDA Trend Breakout</span><br>
        <span style="color:#f1f5f9;">• Entry: Buy on breakout above $190 with volume confirmation</span><br>
        <span style="color:#f1f5f9;">• Stop Loss: $182 (below EMA10 $181.76)</span><br>
        <span style="color:#f1f5f9;">• Target: $200 (R1 pivot $187, R2 $200)</span><br>
        <span style="color:#f1f5f9;">• Risk: ~$8/share → 1 share = $8 risk (within $10 budget)</span><br>
        <span style="color:#f1f5f9;">• Position Size: 1 share @ ~$190 = $190</span><br>
        <span style="color:#34d399;font-weight:600;">Why: All 14 MAs say Buy, MACD positive, RSI 62 = not overbought, clean uptrend</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="glass-card-glow">
        <span style="font-size:1rem;font-weight:700;color:#22d3ee;">STRATEGY 2: AAPL Pullback Buy</span><br>
        <span style="color:#f1f5f9;">• Entry: Buy at $252-$255 (EMA200 support zone)</span><br>
        <span style="color:#f1f5f9;">• Stop Loss: $248 (below EMA200 $251.69)</span><br>
        <span style="color:#f1f5f9;">• Target: $265 (R1 pivot $265)</span><br>
        <span style="color:#f1f5f9;">• Risk: ~$5/share → 2 shares = $10 risk (within budget)</span><br>
        <span style="color:#f1f5f9;">• Position Size: 1 share @ ~$255 = $255 (keeps capital flexible)</span><br>
        <span style="color:#34d399;font-weight:600;">Why: RSI 54 = room to run, bouncing off EMA200, 11/15 MAs bullish</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="glass-card-glow">
        <span style="font-size:1rem;font-weight:700;color:#22d3ee;">STRATEGY 3: INTC Mean Reversion (WAIT)</span><br>
        <span style="color:#f1f5f9;">• Wait for RSI to cool from 78 to ~60</span><br>
        <span style="color:#f1f5f9;">• Entry: Buy pullback to EMA20 ($52) area</span><br>
        <span style="color:#f1f5f9;">• Stop Loss: $48 (below EMA30 $50)</span><br>
        <span style="color:#f1f5f9;">• Target: $65+ (current level = was resistance)</span><br>
        <span style="color:#f1f5f9;">• Risk: ~$4/share → 2 shares = $8 risk</span><br>
        <span style="color:#fbbf24;font-weight:600;">Why: Overbought RSI 78 = likely pullback. Excellent setup AFTER correction. All MAs bullish long-term.</span>
    </div>""", unsafe_allow_html=True)

    # ── Key Rules ──
    st.markdown("#### 📋 Trading Rules Based on Analysis")
    rules = [
        "Never chase overbought stocks (RSI > 70) — wait for pullback",
        "NVDA is the cleanest setup — all MAs aligned, moderate RSI",
        "PLTR is a SELL signal — avoid until trend reverses above EMA50",
        "SOFI bouncing but still in downtrend — need close above $18.50",
        "SPY uptrend = green light for longs, use Trend Breakout strategy",
        "Risk max $10/trade with $500 account (2% rule)",
        "Only trade 1-2 positions at a time to preserve capital",
        "Set stop loss BEFORE entering — never trade without an exit plan",
    ]
    for i, rule in enumerate(rules, 1):
        st.markdown(f'<div class="rule-item"><strong>#{i}</strong> &nbsp; {rule}</div>', unsafe_allow_html=True)

    # ── TradingView Live Chart Widget ──
    st.markdown("#### 📊 Live TradingView Chart")
    tv_symbol = st.selectbox("Select symbol to chart", list(_tv_data.keys()), key="tv_chart_sym")

    exchange_map = {"SPY": "AMEX", "AAPL": "NASDAQ", "NVDA": "NASDAQ", "AMD": "NASDAQ",
                    "INTC": "NASDAQ", "PLTR": "NASDAQ", "SOFI": "NASDAQ", "META": "NASDAQ"}
    tv_exchange = exchange_map.get(tv_symbol, "NASDAQ")

    import streamlit.components.v1 as components
    components.html(f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart" style="height:100%;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_exchange}:{tv_symbol}",
        "interval": "D",
        "timezone": "America/New_York",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#0b1120",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies", "MASimple@tv-basicstudies"],
        "container_id": "tradingview_chart",
        "hide_side_toolbar": false,
        "details": true,
        "hotlist": true,
        "calendar": true
      }});
      </script>
    </div>
    <!-- TradingView Widget END -->
    """, height=520)

    # ── TradingView Technical Analysis Widget ──
    st.markdown("#### 📈 Technical Analysis Gauge")
    components.html(f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "1D",
        "width": "100%",
        "isTransparent": true,
        "height": "450",
        "symbol": "{tv_exchange}:{tv_symbol}",
        "showIntervalTabs": true,
        "locale": "en",
        "colorTheme": "dark"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """, height=470)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Educational app. No strategy works perfectly in live markets. Always manage risk first.")
