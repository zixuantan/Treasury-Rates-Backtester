import pandas as pd
import pytest

from rates_backtester.portfolio import build_daily_portfolio, summarize_pnl_breakdown, summarize_portfolio
from rates_backtester.simulation import TradeResult
from rates_backtester.types import TradeLeg, TradeTemplate


def test_portfolio_metrics():
    trades = [
        TradeResult(
            signal_name="a",
            template=TradeTemplate.OUTRIGHT,
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-03"),
            pnl=1.0,
            gross_notional=1.0,
            legs=(TradeLeg(tenor="10Y", weight=1.0, duration=8.0),),
            metadata={},
        ),
        TradeResult(
            signal_name="b",
            template=TradeTemplate.OUTRIGHT,
            entry_date=pd.Timestamp("2024-01-02"),
            exit_date=pd.Timestamp("2024-01-04"),
            pnl=-0.5,
            gross_notional=1.0,
            legs=(TradeLeg(tenor="10Y", weight=1.0, duration=8.0),),
            metadata={},
        ),
    ]
    frame = pd.DataFrame(
        {"10Y": [4.0, 4.0, -8.5, 10.25]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    daily = build_daily_portfolio(frame, trades, starting_capital=100.0)
    metrics = summarize_portfolio(trades, daily)
    assert metrics.trade_count == 2
    assert metrics.cumulative_pnl == 0.5
    assert metrics.win_rate == 0.5


def test_drawdown_includes_initial_zero_baseline():
    trade = TradeResult(
        signal_name="loss",
        template=TradeTemplate.OUTRIGHT,
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-01-02"),
        pnl=-100.0,
        gross_notional=1_000_000.0,
        legs=(TradeLeg(tenor="10Y", weight=1.0, duration=8.0),),
        metadata={},
    )
    frame = pd.DataFrame(
        {"10Y": [4.0, 4.00125]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    daily = build_daily_portfolio(frame, [trade], starting_capital=1_000.0)
    assert summarize_portfolio([trade], daily).max_drawdown == pytest.approx(-100.0)


def test_pnl_breakdown_groups_trades_and_calculates_payoff_statistics():
    trades = [
        TradeResult(
            signal_name="yield_rally_momentum",
            template=TradeTemplate.OUTRIGHT,
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-03"),
            pnl=200.0,
            gross_notional=1_000_000.0,
            legs=(TradeLeg(tenor="10Y", weight=1.0, duration=8.0),),
            metadata={},
        ),
        TradeResult(
            signal_name="yield_rally_momentum",
            template=TradeTemplate.OUTRIGHT,
            entry_date=pd.Timestamp("2025-01-01"),
            exit_date=pd.Timestamp("2025-01-03"),
            pnl=-100.0,
            gross_notional=1_000_000.0,
            legs=(TradeLeg(tenor="10Y", weight=1.0, duration=8.0),),
            metadata={},
        ),
    ]
    breakdown = summarize_pnl_breakdown(trades)
    assert breakdown.average_winner == 200.0
    assert breakdown.average_loser == -100.0
    assert breakdown.profit_factor == 2.0
    assert breakdown.by_category[0].label == "directional yield"
    assert breakdown.by_category[0].cumulative_pnl == 100.0
    assert [row.label for row in breakdown.by_exit_year] == ["2024", "2025"]
