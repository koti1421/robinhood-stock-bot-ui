from __future__ import annotations

import os
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.backtest import BacktestConfig, run_backtest
from src.data import fetch_many, fetch_prices
from src.risk import position_size
from src.robinhood_client import RobinhoodClient, RobinhoodOrderRequest, env_credentials
from src.strategy import StrategyConfig, market_regime_on, scan_universe

load_dotenv()

st.set_page_config(page_title="US Stock Bot UI", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem;}
.hero {
    background: linear-gradient(120deg, #0B1F42 0%, #0047FF 60%, #00B2FF 100%);
    border-radius: 16px;
    padding: 20px 24px;
    color: white;
    margin-bottom: 1rem;
}
.metric-card {
    border: 1px solid #E4E7EC;
    border-radius: 14px;
    padding: 0.6rem;
    background: #FFFFFF;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h2 style="margin:0">US Stocks Strategy Dashboard</h2><p style="margin:0.4rem 0 0">Trend breakout + risk controls + Robinhood execution guardrails</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    symbols_input = st.text_area(
        "Universe symbols (comma-separated)",
        value="AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,JPM,XOM,AVGO",
    )
    equity = st.number_input("Account Equity ($)", value=25000.0, min_value=1000.0, step=1000.0)
    risk_per_trade = st.slider("Risk Per Trade", min_value=0.001, max_value=0.02, value=0.005, step=0.001)

symbols: List[str] = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
spy_df = fetch_prices("SPY", years=3)
regime = market_regime_on(spy_df)

c1, c2, c3 = st.columns(3)
c1.metric("Market Regime", "ON" if regime else "OFF")
c2.metric("Universe Size", f"{len(symbols)}")
c3.metric("Risk/Trade", f"{risk_per_trade*100:.2f}%")

tab1, tab2, tab3 = st.tabs(["Signal Scanner", "Backtest", "Robinhood Orders"])

with tab1:
    st.subheader("Daily Signal Scanner")
    if st.button("Scan Now", type="primary"):
        with st.spinner("Downloading prices and scanning..."):
            prices = fetch_many(symbols, years=3)
            rows = scan_universe(prices, spy_df, StrategyConfig())

        if not regime:
            st.warning("SPY is below 200 SMA. New long entries are disabled by regime filter.")
        if not rows:
            st.info("No breakout signals today.")
        else:
            out = []
            for row in rows:
                stop = row["close"] - 2.0 * row["atr"]
                qty = position_size(equity, risk_per_trade, row["close"], stop)
                out.append(
                    {
                        "Symbol": row["symbol"],
                        "Close": round(row["close"], 2),
                        "ATR": round(row["atr"], 2),
                        "Stop": round(stop, 2),
                        "Qty": qty,
                    }
                )
            st.dataframe(pd.DataFrame(out), use_container_width=True)

with tab2:
    st.subheader("Backtest")
    bt_symbol = st.text_input("Ticker", value="AAPL").upper().strip()
    fee_bps = st.slider("Slippage + Fees (bps per side)", 0, 30, 5)

    if st.button("Run Backtest"):
        bt_df = fetch_prices(bt_symbol, years=5)
        if bt_df.empty or len(bt_df) < 300:
            st.error("Not enough data for backtest.")
        else:
            curve, stats = run_backtest(bt_df, BacktestConfig(fee_bps=fee_bps))
            m1, m2, m3 = st.columns(3)
            m1.metric("Strategy Return", f"{stats['Strategy Return']*100:.2f}%")
            m2.metric("Buy/Hold Return", f"{stats['Buy/Hold Return']*100:.2f}%")
            m3.metric("Max Drawdown", f"{stats['Max Drawdown']*100:.2f}%")

            chart_df = curve[["equity_curve", "buy_hold"]].reset_index().rename(columns={"index": "Date"})
            fig = px.line(chart_df, x="Date", y=["equity_curve", "buy_hold"], title=f"{bt_symbol} Backtest")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Robinhood Execution")
    st.caption("Keep PAPER_MODE=true until your paper and live checks are fully validated.")

    env_user, env_pass = env_credentials()
    default_user = env_user if env_user else ""
    default_pass = env_pass if env_pass else ""

    rh_user = st.text_input("Robinhood Username", value=default_user)
    rh_pass = st.text_input("Robinhood Password", value=default_pass, type="password")
    mfa_code = st.text_input("MFA Code (if prompted)")
    paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"

    if "rh_client" not in st.session_state:
        st.session_state["rh_client"] = RobinhoodClient()

    if st.button("Login Robinhood"):
        try:
            ok = st.session_state["rh_client"].login(rh_user, rh_pass, mfa_code=mfa_code or None)
            if ok:
                buying_power = st.session_state["rh_client"].profile_buying_power()
                st.success(f"Logged in. Buying power: ${buying_power:,.2f}")
            else:
                st.error("Login failed.")
        except Exception as exc:
            st.error(f"Login error: {exc}")

    st.markdown("---")
    st.write("Manual Order Ticket")
    o1, o2, o3 = st.columns(3)
    order_symbol = o1.text_input("Symbol", value="AAPL").upper().strip()
    side = o2.selectbox("Side", ["buy", "sell"])
    qty = o3.number_input("Quantity", min_value=1, value=1, step=1)
    limit_price = st.number_input("Limit Price (0 for market)", min_value=0.0, value=0.0, step=0.01)

    if st.button("Submit Order", type="primary"):
        try:
            req = RobinhoodOrderRequest(
                symbol=order_symbol,
                side=side,
                quantity=int(qty),
                limit_price=float(limit_price) if limit_price > 0 else None,
            )
            result = st.session_state["rh_client"].submit_order(req, paper_mode=paper_mode)
            if paper_mode:
                st.warning("PAPER_MODE is enabled. No live order was sent.")
            st.json(result)
        except Exception as exc:
            st.error(f"Order error: {exc}")

st.markdown("---")
st.caption("Educational app. No strategy works perfectly in live markets.")
