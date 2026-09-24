# Backtester Spec Audit

## Metrics

- Present: `spread_2s10s`, `spread_5s30s`, `fly_2s5s10s`, `z_2s10s`, `z_5s30s`, `z_butterfly`, `y10_change_1d`, `y10_change_5d`, `y10_change_20d`, `z_y10_change_5d`, `z_y10_change_20d`, `cpi_surprise`, `nfp_surprise`, `z_cpi_surprise`, `z_nfp_surprise`
- Added: `add_required_metrics()` now accepts the raw `date/y2/y5/y10/y30` input shape via aliasing and computes the full metric set with no-lookahead rolling statistics.
- Still missing: None for the checked metrics layer.

## Signals

- Present: `hot_CPI`, `cold_CPI`, `strong_payrolls`, `weak_payrolls`, `yield_selloff_momentum`, `yield_rally_momentum`, `yield_selloff_exhaustion`, `yield_rally_exhaustion`, `curve_too_flat`, `curve_too_steep`, `long_end_too_flat`, `long_end_too_steep`, `five_y_cheap`, `five_y_rich`
- Added: `SIGNAL_RULES` and `generate_signal_events()` now emit `date`, `signal_name`, and `interpretation`.
- Still missing: None for the checked signals layer.

## Signal-to-trade mappings

- Present: All required mappings in `SIGNAL_TO_TRADE`.
- Added: Explicit `SignalTradeMapping` and `TradeLegSpec` objects with long/short side labels and relative weights. Simulation normalizes those weights to the configured gross dollar notional.
- Still missing: None for the checked mapping layer.

## Directional sanity checks

- Long duration profits when yields fall: Verified via an outright long trade in simulation.
- Short duration profits when yields rise: Verified via an outright short trade in simulation.
- 2s10s steepener legs: `long 2Y`, `short 10Y`
- 2s10s flattener legs: `short 2Y`, `long 10Y`
- 5s30s steepener legs: `short 5Y`, `long 30Y`
- 5s30s flattener legs: `long 5Y`, `short 30Y`
- 5Y cheap butterfly legs: `long 5Y`, `short 2Y`, `short 10Y`
- 5Y rich butterfly legs: `short 5Y`, `long 2Y`, `long 10Y`

## Files changed

- `src/rates_backtester/data.py`
- `src/rates_backtester/features.py`
- `src/rates_backtester/signals.py`
- `src/rates_backtester/trades.py`
- `src/rates_backtester/__init__.py`
- `tests/test_features.py`
- `tests/test_spec_audit.py`
- `SPEC_AUDIT.md`

## Verification

- Test suite status: `30 passed`
- No static errors reported in the edited source and test files.

The reduced test count reflects removal of the duplicate legacy signal/simulation API and its redundant tests, not a loss of coverage for the production CLI path.
