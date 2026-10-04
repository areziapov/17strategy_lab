# Strategy Lab v1.4

## New: Strategy Tester

A fourth workflow is now included: **Strategy Tester**.

It lets you set:
- Long or Short
- Nested entry conditions
- Nested exit conditions
- Bull / Bear / Transition regimes
- Starting capital
- Capital allocation per trade
- Leverage
- Stop loss
- Trading fees/slippage assumption from the sidebar
- Date range
- Optional simplified liquidation proxy

## Portfolio math

Position margin = account equity × allocation %

Position notional = margin × leverage

PnL = notional × underlying trade return − notional × round-trip cost

The next trade compounds from the resulting account equity.

## Stop loss

The stop is an underlying-price percentage from entry.
It is checked intrabar using OHLC.

If a 4H bar gaps beyond the stop, the simulator uses the bar open when it is worse than the stop level.

## Liquidation proxy

Optional and deliberately simplified:

adverse underlying move ≈ 1 / leverage

This does **not** model a specific exchange's maintenance-margin formula.
It is there to prevent obviously impossible tests such as 20x leverage with a 15% stop.

## Outputs

- Ending capital
- Total account return
- Max account-equity drawdown
- Number of trades
- Win rate
- Stop exits
- Liquidation-proxy exits
- Equity curve
- Performance by market regime
- Performance by year
- Complete trade history
- MFE / MAE / per-trade max drawdown
- Capital before/after each trade, notional, leverage, fees and net PnL

## Data cutoff

Hard cutoff remains 19 Apr 2026.

## Run

```bash
cd ~/Downloads/strategy_lab_v1_4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
