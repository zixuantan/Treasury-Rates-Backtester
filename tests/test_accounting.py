import pandas as pd
import pytest

from rates_backtester import (
    BacktestConfig,
    BacktestRun,
    SignalEvent,
    backtest_run_to_dict,
    build_daily_portfolio,
    simulate_signal_events,
    summarize_portfolio,
)
from rates_backtester.__main__ import _print_summary


def _falling_ten_year_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "2Y": [4.0, 4.0],
            "5Y": [4.0, 4.0],
            "10Y": [4.0, 3.9],
            "30Y": [4.0, 4.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )


def _rally_event(date: pd.Timestamp) -> SignalEvent:
    return SignalEvent(
        date=date,
        signal_name="yield_rally_momentum",
        interpretation="test",
        metric_name="z_y10_change_5d",
        metric_value=-2.0,
        threshold=-1.0,
    )


def test_percentage_point_yields_are_converted_to_decimal_pnl():
    frame = _falling_ten_year_frame()
    trades = simulate_signal_events(
        frame,
        [_rally_event(frame.index[0])],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=1_000_000),
    )
    assert trades[0].pnl == pytest.approx(8_000.0)


def test_short_bond_profits_when_yield_rises():
    frame = _falling_ten_year_frame().copy()
    frame.loc[frame.index[1], "10Y"] = 4.1
    event = SignalEvent(
        date=frame.index[0],
        signal_name="yield_selloff_momentum",
        interpretation="test",
        metric_name="z_y10_change_5d",
        metric_value=2.0,
        threshold=1.0,
    )
    trades = simulate_signal_events(
        frame,
        [event],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=1_000_000),
    )
    assert trades[0].pnl == pytest.approx(8_000.0)


def test_configured_notional_scales_pnl_and_leg_weights_are_normalized():
    frame = _falling_ten_year_frame()
    trades = simulate_signal_events(
        frame,
        [_rally_event(frame.index[0])],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=2_000_000),
    )
    assert trades[0].pnl == pytest.approx(16_000.0)
    assert sum(abs(leg.weight) for leg in trades[0].legs) == pytest.approx(1.0)


def test_daily_portfolio_marks_open_trade_to_market_and_calculates_return():
    frame = _falling_ten_year_frame()
    trades = simulate_signal_events(
        frame,
        [_rally_event(frame.index[0])],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=1_000_000),
    )
    daily = build_daily_portfolio(frame, trades, starting_capital=10_000_000.0)

    assert daily["daily_pnl"].tolist() == pytest.approx([0.0, 8_000.0])
    assert daily["daily_return"].tolist() == pytest.approx([0.0, 0.0008])
    assert daily["cumulative_return"].iloc[-1] == pytest.approx(0.0008)


def test_overlap_and_cooldown_controls_skip_entries():
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    frame = pd.DataFrame(
        {tenor: [4.0] * len(index) for tenor in ("2Y", "5Y", "10Y", "30Y")},
        index=index,
    )
    events = [_rally_event(index[position]) for position in (0, 1, 4)]
    constrained = simulate_signal_events(
        frame,
        events,
        BacktestConfig(
            signal_lag=0,
            holding_period=2,
            cooldown_period=2,
        ),
    )

    assert len(constrained) == 1


def test_backtest_json_serializes_materialized_trade_legs():
    frame = _falling_ten_year_frame()
    event = _rally_event(frame.index[0])
    trades = simulate_signal_events(
        frame,
        [event],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=1_000_000),
    )
    daily = build_daily_portfolio(frame, trades, starting_capital=10_000_000.0)
    result = BacktestRun(frame, [event], trades, daily, summarize_portfolio(trades, daily))
    payload = backtest_run_to_dict(result)
    assert payload["period"] == {
        "data_start": "2024-01-01T00:00:00",
        "data_end": "2024-01-02T00:00:00",
        "first_trade_entry": "2024-01-01T00:00:00",
        "last_trade_exit": "2024-01-02T00:00:00",
    }
    assert payload["pnl_breakdown"]["by_category"][0]["label"] == "directional yield"
    assert payload["trades"][0]["legs"][0] == {
        "tenor": "10Y",
        "side": "long",
        "weight": 1.0,
        "duration": 8.0,
    }


def test_default_summary_prints_period_and_pnl_breakdowns(capsys):
    frame = _falling_ten_year_frame()
    event = _rally_event(frame.index[0])
    trades = simulate_signal_events(
        frame,
        [event],
        BacktestConfig(signal_lag=0, holding_period=1, gross_notional=1_000_000),
    )
    daily = build_daily_portfolio(frame, trades, starting_capital=10_000_000.0)
    _print_summary(BacktestRun(frame, [event], trades, daily, summarize_portfolio(trades, daily)))
    output = capsys.readouterr().out
    assert "data: 2024-01-01 to 2024-01-02" in output
    assert "completed trades: 2024-01-01 to 2024-01-02" in output
    assert "P&L by signal family" in output
    assert "P&L by signal" in output
    assert "P&L by exit year" in output
