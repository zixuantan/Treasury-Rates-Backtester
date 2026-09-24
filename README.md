# Treasury Rates Backtester

A small, modular Python backtester for Treasury yield-curve and relative-value trades.

## Scope

- Daily US Treasury yields for 2Y, 5Y, 10Y, and 30Y tenors
- Curve spreads, butterflies, momentum, and rolling z-scores
- Fixed-decay Nelson-Siegel level, slope, curvature, and fit-error factors
- Optional breakeven, broad-dollar, credit-spread, SPY, and VIX context
- Declarative signal rules mapped to trade templates
- Optional conflict-veto regime filters using risk sentiment, inflation context, and Nelson–Siegel factors
- Trading-day simulation with duration-based approximate dollar P&L
- Portfolio metrics and trade logs

## Data source

- Raw Treasury inputs live in `Treasury Yield/` as `DGS2.csv`, `DGS5.csv`, `DGS10.csv`, and `DGS30.csv`
- `load_dataset("Treasury Yield")` merges those files into a single wide frame with `date`, `y2`, `y5`, `y10`, and `y30`
- Raw risk-sentiment inputs live in `Risk Sentiment Metrics/` as `SP500.csv` and `VIXCLS.csv`
- `load_dataset("Risk Sentiment Metrics")` merges those files into a single wide frame with `date`, `SPY`, and `VIX`
- `load_dataset(".")` merges both folders when they are present in the workspace root
- Optional FRED-format context files can live in `Market Context/` as `T5YIE.csv`, `T10YIE.csv`, `T5YIFR.csv`, `DTWEXBGS.csv`, `BAMLH0A0HYM2.csv`, and `BAMLC0A0CM.csv`
- Optional 1M, 3M, 6M, and 1Y Treasury files are automatically included when present alongside the required Treasury files

## Design choices

- Canonical input is a wide dataframe with one row per trading date.
- Signals are evaluated without lookahead by shifting feature inputs.
- Multiple positions can be open at once.
- Regime filters are enabled by default. Missing or neutral context does not block a trade; clearly contradictory context does.
- P&L uses a yield-change approximation, not full bond pricing.
- Treasury yields expressed in percentage points are converted to decimal changes before P&L is calculated.
- `gross_notional` is the total absolute dollar notional allocated across a trade's legs.
- Ordinary FRED downloads are current revised history. Revisable macro data require vintage-aware release data before use in a historical strategy.

## How the backtest works

The backtest follows this sequence:

```text
Input data
  -> calculate features and rolling z-scores
  -> generate signals whenever threshold conditions are met
  -> map each signal to a duration, curve, or butterfly trade
  -> enter on the next trading row
  -> hold for five trading days by default
  -> exit and calculate approximate P&L
  -> aggregate portfolio metrics
```

The main active signal families are:

- **Duration:** long or short 10Y momentum and reversal trades
- **Curve:** 2s10s and 5s30s steepeners or flatteners
- **Butterfly:** long or short 2s5s10s relative-value trades

CPI and payroll signal definitions are present, but they remain inactive without timestamped first-release actual and pre-release consensus data.

Signals are evaluated independently on every date. If a condition remains beyond its threshold for several consecutive days, the engine opens a new trade each day. It does not currently require a fresh threshold crossing, impose a cooldown, or prevent an equivalent trade from already being open.

For example:

```text
Monday:    signal is true -> enter trade A on Tuesday
Tuesday:   signal is true -> enter trade B on Wednesday
Wednesday: signal is true -> enter trade C on Thursday
```

All three trades can remain open simultaneously until their individual five-day holding periods end.

## Position sizing

The default gross notional is $1,000,000 per trade. For a multi-leg trade, gross notional means:

```text
sum(abs(leg notionals)) = $1,000,000
```

The engine uses approximate durations of 1.9, 4.5, 8.0, and 18.0 for the 2Y, 5Y, 10Y, and 30Y tenors respectively. Curve and butterfly weights are chosen so opposing legs have approximately offsetting dollar-duration exposure.

For a 2s10s steepener, the trade is long 2Y and short 10Y. The relative 10Y weight is:

```text
2Y duration / 10Y duration = 1.9 / 8.0 = 0.2375
```

After normalizing the absolute weights to one, a $1,000,000 gross trade is approximately:

```text
Long  $808,081 of 2Y
Short $191,919 of 10Y
```

The two legs then have similar DV01. A parallel yield move should approximately cancel, while a change in the 2s10s spread drives P&L.

For a 5Y-cheap butterfly, the trade is long the 5Y belly and short the 2Y and 10Y wings. Approximate normalized allocations are:

```text
Long  40.6% of gross notional in 5Y
Short 48.0% in 2Y
Short 11.4% in 10Y
```

The combined duration exposure of the two wings approximately offsets the belly. These are first-order duration hedges, not complete hedges against convexity, carry, rolldown, or non-parallel curve movements.

## P&L calculation

Each leg uses the duration approximation:

```text
leg P&L = -gross notional x signed weight x duration x yield change
```

A positive weight is long and a negative weight is short. Because the input yields are quoted in percentage points, the engine divides their difference by 100 before applying the formula.

For example, suppose a $1,000,000 long 10Y trade has duration 8 and its yield falls from 4.00% to 3.90%:

```text
yield change = (3.90 - 4.00) / 100 = -0.001
P&L = -$1,000,000 x 1.0 x 8 x -0.001
    = +$8,000
```

The long bond profits because its yield fell. A short position with the same move would lose approximately $8,000.

Gross notional applies to every newly opened trade rather than to the portfolio as a whole. Three overlapping signals can therefore create three separate $1,000,000 gross positions. The current engine does not impose a portfolio-wide gross-exposure limit or net overlapping positions.

## Package layout

- `src/rates_backtester/data.py`
- `src/rates_backtester/features.py`
- `src/rates_backtester/signals.py`
- `src/rates_backtester/trades.py`
- `src/rates_backtester/simulation.py`
- `src/rates_backtester/portfolio.py`
- `src/rates_backtester/types.py`
- `src/rates_backtester/market_data.py`
- `src/rates_backtester/nelson_siegel.py`

## Tests

Run `pytest` from the repository root.

## Command-line usage

After installing the package in editable mode with `pip install -e .`:

```bash
python -m rates_backtester .
```

Or, after installation, use the console script directly:

```bash
rates-backtester .
```

To print the full result, including materialized trade legs, as JSON:

```bash
python -m rates_backtester . --json
```

The default terminal report includes:

- Full input-data date range and effective completed-trade date range
- Cumulative P&L, Sharpe ratio, maximum drawdown, win rate, and trade count
- Average winning and losing trade plus profit factor
- P&L, average P&L, win rate, and trade count by strategy category
- The same breakdown by individual signal and exit year

The JSON output contains the same information under `period`, `portfolio`, and `pnl_breakdown`, followed by the complete signal and trade records.

## Streamlit dashboard

Install the dashboard dependency and launch the local research UI:

```bash
pip install -e '.[dashboard]'
streamlit run streamlit_app.py
```

The Treasury Rates Backtester dashboard imports the same package pipeline used by the CLI. It includes headline metrics and test dates, filtered-versus-unfiltered performance and equity-curve comparisons, strategy-level filter impact, accepted/rejected signal diagnostics, an auditable trade log, Treasury curve history, regime scores, and Nelson–Siegel factors. Its Methodology tab defines every performance metric, documents every signal threshold and trade mapping, and explains position sizing and approximate P&L. Controls in the sidebar allow the lookback, holding period, entry lag, annualization factor, gross notional, and displayed filter mode to be changed without duplicating strategy code.

Regime filters are enabled by default and can be switched off in the sidebar or with `--no-regime-filters` on the CLI. Directional-duration candidates are vetoed when the combined SPY, VIX, available credit-spread, broad-dollar, and breakeven regime conflicts with the trade direction. Nelson–Siegel level, slope, and curvature can veto conflicting momentum/exhaustion, curve, and butterfly candidates respectively. The Signals tab reports every candidate and its accept/reject reason.

## Example

```python
from rates_backtester import load_dataset, run_backtest

frame = load_dataset("rates.csv")
result = run_backtest(frame, rolling_window=252)
print(result.portfolio)
```

## Optional FRED market context

Set `FRED_API_KEY`, then fetch and freeze the daily market-context inputs:

```python
from datetime import date

from rates_backtester import (
    FREDClient,
    MARKET_CONTEXT_SERIES,
    export_fred_csv_directory,
)

client = FREDClient()
result = client.get_series(
    MARKET_CONTEXT_SERIES,
    date(2021, 1, 1),
    date.today(),
)
export_fred_csv_directory(result.data, "Market Context")
```

The root dataset loader will merge that folder on the Treasury trading dates. Signals observed through a market close are entered on the next trading row by default.

`FREDClient.get_series(..., vintage_date=...)` can request the observation history known on one specific date. This is useful for policy snapshots, but it is not a substitute for a complete first-release/consensus dataset.

## Additional datasets needed

The project can run with Treasury yields plus SPY/VIX. The remaining datasets are deferred to a later phase:

- CPI actual and consensus time series for `cpi_actual` and `cpi_consensus`
- Non-farm payrolls actual and consensus time series for `nfp_actual` and `nfp_consensus`
- Optional market context series for `DXY`

`DTWEXBGS` is supported as a broad trade-weighted dollar index. It should not be described as the ICE DXY index.

## Modeling boundaries

- CPI and payroll surprise rules remain inactive without timestamped first-release actual and pre-release consensus data.
- Optional market-context fields currently enrich the research frame; they are not mapped to standalone trades.
- Ordinary FRED macro histories can contain revisions and must not be treated as point-in-time observations.
- P&L is a first-order duration approximation and excludes convexity, carry, rolldown, coupons, financing, transaction costs, and slippage.
- Portfolio P&L is recognized on trade exit dates rather than marked to market each day.

## Next steps

1. Freeze the desired FRED market-context series into `Market Context/` for reproducible runs.
2. Define and validate trading rules that use breakeven, dollar, credit, or Nelson-Siegel factors.
3. Add point-in-time CPI and payroll release/consensus data before enabling macro-surprise strategies.
4. Add daily mark-to-market accounting and execution costs before interpreting production-level performance.
