from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import pandas as pd

from .simulation import TradeResult


@dataclass(frozen=True)
class PortfolioMetrics:
    cumulative_pnl: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int


@dataclass(frozen=True)
class PnlBreakdownRow:
    label: str
    trade_count: int
    cumulative_pnl: float
    average_pnl: float
    win_rate: float


@dataclass(frozen=True)
class PnlBreakdown:
    average_winner: float | None
    average_loser: float | None
    profit_factor: float | None
    by_category: tuple[PnlBreakdownRow, ...]
    by_signal: tuple[PnlBreakdownRow, ...]
    by_exit_year: tuple[PnlBreakdownRow, ...]


def _strategy_category(trade: TradeResult) -> str:
    signal = trade.signal_name
    if "momentum" in signal or "exhaustion" in signal:
        return "directional yield"
    if signal.startswith(("curve_", "long_end_")):
        return "curve"
    if signal.startswith("five_y_"):
        return "butterfly"
    if signal in {"hot_CPI", "cold_CPI", "strong_payrolls", "weak_payrolls"}:
        return "macro surprise"
    return trade.template.value


def _group_breakdown(
    trades: Sequence[TradeResult],
    labels: Sequence[str],
    *,
    chronological: bool = False,
) -> tuple[PnlBreakdownRow, ...]:
    grouped: dict[str, list[TradeResult]] = {}
    for trade, label in zip(trades, labels):
        grouped.setdefault(label, []).append(trade)
    rows = [
        PnlBreakdownRow(
            label=label,
            trade_count=len(group),
            cumulative_pnl=float(sum(trade.pnl for trade in group)),
            average_pnl=float(sum(trade.pnl for trade in group) / len(group)),
            win_rate=float(sum(trade.pnl > 0 for trade in group) / len(group)),
        )
        for label, group in grouped.items()
    ]
    if chronological:
        rows.sort(key=lambda row: row.label)
    else:
        rows.sort(key=lambda row: (row.cumulative_pnl, row.label))
    return tuple(rows)


def summarize_pnl_breakdown(trades: Sequence[TradeResult]) -> PnlBreakdown:
    """Summarize realized trade P&L by category, signal, and exit year."""
    winners = [trade.pnl for trade in trades if trade.pnl > 0]
    losers = [trade.pnl for trade in trades if trade.pnl < 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    return PnlBreakdown(
        average_winner=float(sum(winners) / len(winners)) if winners else None,
        average_loser=float(sum(losers) / len(losers)) if losers else None,
        profit_factor=float(gross_profit / gross_loss) if gross_loss else None,
        by_category=_group_breakdown(trades, [_strategy_category(trade) for trade in trades]),
        by_signal=_group_breakdown(trades, [trade.signal_name for trade in trades]),
        by_exit_year=_group_breakdown(
            trades,
            [str(trade.exit_date.year) for trade in trades],
            chronological=True,
        ),
    )


def build_daily_portfolio(
    frame: pd.DataFrame,
    trades: Sequence[TradeResult],
    *,
    starting_capital: float,
    evaluation_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Mark open trades to market each day using the duration approximation."""
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if frame.empty:
        return pd.DataFrame(
            columns=["daily_pnl", "portfolio_value", "daily_return", "cumulative_return"]
        )

    start = pd.Timestamp(evaluation_start) if evaluation_start is not None else pd.Timestamp(frame.index.min())
    dates = frame.index[frame.index >= start]
    daily_pnl = pd.Series(0.0, index=dates, dtype=float)
    yield_changes = frame[[column for column in ("2Y", "5Y", "10Y", "30Y") if column in frame]].diff() / 100.0

    for trade in trades:
        active_dates = dates[(dates > trade.entry_date) & (dates <= trade.exit_date)]
        for leg in trade.legs:
            daily_pnl.loc[active_dates] += (
                -trade.gross_notional
                * leg.weight
                * leg.duration
                * yield_changes.loc[active_dates, leg.tenor]
            )

    portfolio_value = starting_capital + daily_pnl.cumsum()
    previous_value = portfolio_value.shift(1, fill_value=starting_capital)
    if (previous_value <= 0).any() or (portfolio_value <= 0).any():
        raise ValueError("starting_capital is too small for the simulated portfolio losses")
    daily_return = daily_pnl / previous_value
    return pd.DataFrame(
        {
            "daily_pnl": daily_pnl,
            "portfolio_value": portfolio_value,
            "daily_return": daily_return,
            "cumulative_return": portfolio_value / starting_capital - 1.0,
        }
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    # Include the pre-trade zero baseline so an immediate loss is measured as
    # drawdown rather than becoming the initial running maximum.
    values = pd.concat([pd.Series([0.0]), equity_curve.reset_index(drop=True)], ignore_index=True)
    running_max = values.cummax()
    drawdown = values - running_max
    return float(drawdown.min())


def summarize_portfolio(
    trades: Sequence[TradeResult],
    daily_portfolio: pd.DataFrame,
    annualization_factor: int = 252,
) -> PortfolioMetrics:
    if daily_portfolio.empty:
        return PortfolioMetrics(0.0, 0.0, 0.0, 0.0, 0.0, len(trades))
    daily_pnl = daily_portfolio["daily_pnl"]
    daily_returns = daily_portfolio["daily_return"]
    cumulative = float(daily_pnl.sum())
    cumulative_return = float(daily_portfolio["cumulative_return"].iloc[-1])
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) != 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1) * sqrt(annualization_factor))
    else:
        sharpe = 0.0
    win_rate = float(sum(1 for trade in trades if trade.pnl > 0) / len(trades)) if trades else 0.0
    portfolio_value = daily_portfolio["portfolio_value"]
    max_dd = _max_drawdown(portfolio_value - float(portfolio_value.iloc[0]))
    return PortfolioMetrics(cumulative, cumulative_return, sharpe, max_dd, win_rate, len(trades))
