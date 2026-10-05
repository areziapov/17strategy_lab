# Strategy Lab v1.5

Updated with the new Telegram export after 19 Apr 2026.

## Data
- Reconciled master now runs from 2022-01-01 through 2026-10-04 10:00 (TradingView 4H base).
- New Telegram signal messages were parsed after the previous hard cutoff.
- Existing history through 19 Apr 2026 was preserved exactly; it was not reinterpreted.
- New Telegram events extend through 2026-10-02 16:59:10.

## Research periods
The app now tags rows/trades as:
- development_2022_2024
- validation_2025
- preupdate_2026_to_apr19
- new_oos_after_apr19

The last bucket is especially important: it is genuinely new data that was not available when the earlier strategies were selected.

## Features retained
- Nested AND/OR entry and exit groups
- Per-group memory conditions
- Bull/Bear/Transition filter
- Strategy Search
- Capital Strategy Tester with starting capital, allocation, leverage, stop loss and simplified liquidation proxy
- Per-trade MFE, MAE and max drawdown
- Performance by regime, year and research period

## Run
```bash
cd ~/Downloads/strategy_lab_v1_5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
