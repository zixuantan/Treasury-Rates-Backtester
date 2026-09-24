from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import pandas as pd

from .simulation import TradeResult


@dataclass(frozen=True)
class PortfolioMetrics:
    cumulative_pnl: float
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
    if "momentum" in signal:
        return "momentum"
    if "exhaustion" in signal:
        return "exhaustion"
    if signal.startswith(("curve_", "long_end_")):
        return "curve"
    if signal.startswith("five_y_"):
        return "butterfly"
    if signal in {"hot_CPI", "cold_CPI", "strong_payrolls", "weak_payrolls"}:
        return "macro"
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


def _daily_pnl_series(trades: Sequence[TradeResult]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    records = [(trade.exit_date, trade.pnl) for trade in trades]
    index = pd.DatetimeIndex([exit_date for exit_date, _ in records])
    values = [pnl for _, pnl in records]
    series = pd.Series(values, index=index).sort_index()
    return series.groupby(level=0).sum().sort_index()


def _max_drawdown(equity_curve: pd.Series) -> float:
    # Include the pre-trade zero baseline so an immediate loss is measured as
    # drawdown rather than becoming the initial running maximum.
    values = pd.concat([pd.Series([0.0]), equity_curve.reset_index(drop=True)], ignore_index=True)
    running_max = values.cummax()
    drawdown = values - running_max
    return float(drawdown.min())


def summarize_portfolio(trades: Sequence[TradeResult], annualization_factor: int = 252) -> PortfolioMetrics:
    if not trades:
        return PortfolioMetrics(0.0, 0.0, 0.0, 0.0, 0)
    daily_pnl = _daily_pnl_series(trades)
    cumulative = float(daily_pnl.sum())
    equity_curve = daily_pnl.cumsum()
    if len(daily_pnl) > 1 and daily_pnl.std(ddof=0) != 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std(ddof=0) * sqrt(annualization_factor))
    else:
        sharpe = 0.0
    win_rate = float(sum(1 for trade in trades if trade.pnl > 0) / len(trades))
    max_dd = _max_drawdown(equity_curve) if not equity_curve.empty else 0.0
    return PortfolioMetrics(cumulative, sharpe, max_dd, win_rate, len(trades))
