# US Stocks Bot UI (Robinhood)

A Streamlit app for:
- Daily breakout signal scanning for US stocks
- Basic backtesting with cost assumptions
- Robinhood login and guarded order ticket (paper mode supported)

## Important
This project is educational and does not guarantee profits. Keep `PAPER_MODE=true` until you have validated behavior in your own environment.

## Features
- Strategy: Trend breakout with 50/200 SMA filter and 20-day breakout
- Regime filter: SPY above 200 SMA
- Risk sizing: fixed risk per trade with stop distance
- Backtest: compares strategy vs buy-and-hold
- Robinhood adapter: login + market/limit buy/sell

## Setup
1. Create a virtual environment
2. Install dependencies
3. Copy `.env.example` to `.env`
4. Fill your Robinhood credentials in `.env`
5. Run the app

```bash
cd /Users/km/robinhood-stock-bot-ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Credentials Safety
- Do not share your Robinhood credentials in chat.
- Keep credentials only in local `.env`.
- Use MFA on your account.

## Notes on Robinhood
Robinhood retail access is commonly done through unofficial libraries. API behavior may change. Verify all behavior in paper/small size first.
