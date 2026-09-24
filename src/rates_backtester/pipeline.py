from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import (
    MARKET_CONTEXT_FILES,
    RISK_SENTIMENT_FILES,
    TREASURY_YIELD_FILES,
    canonicalize_yield_frame,
    load_market_context_directory,
    load_risk_sentiment_directory,
    load_treasury_yield_directory,
)
from .features import DEFAULT_ROLLING_WINDOW, add_market_context_metrics, add_required_metrics, rolling_zscore
from .nelson_siegel import add_nelson_siegel_factors
from .portfolio import PortfolioMetrics, PnlBreakdownRow, summarize_pnl_breakdown, summarize_portfolio
from .regimes import RegimeDecision, apply_regime_filters
from .signals import SIGNAL_TO_TRADE, SignalEvent, generate_signal_events
from .trades import DURATION_ESTIMATES, SignalTradeMapping, normalize_gross_weights
from .types import BacktestConfig, TradeLeg, TradeTemplate
from .simulation import TradeResult


@dataclass(frozen=True)
class BacktestRun:
    frame: pd.DataFrame
    signal_events: list[SignalEvent]
    trades: list[TradeResult]
    portfolio: PortfolioMetrics
    regime_decisions: tuple[RegimeDecision, ...] = ()


def load_dataset(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        if read_csv_kwargs:
            raise ValueError("Keyword arguments are only supported for file-based datasets")
        treasury_files = set(TREASURY_YIELD_FILES)
        risk_files = set(RISK_SENTIMENT_FILES)
        market_context_files = set(MARKET_CONTEXT_FILES)
        treasury_dir = dataset_path / "Treasury Yield"
        risk_dir = dataset_path / "Risk Sentiment Metrics"
        market_context_dir = dataset_path / "Market Context"
        if treasury_dir.is_dir():
            treasury_frame = load_treasury_yield_directory(treasury_dir)
            merged = treasury_frame
            if risk_dir.is_dir():
                risk_frame = load_risk_sentiment_directory(risk_dir)
                merged = merged.merge(risk_frame, on="date", how="left")
            if market_context_dir.is_dir():
                market_frame = load_market_context_directory(market_context_dir)
                merged = merged.merge(market_frame, on="date", how="left")
            return merged.sort_values("date").reset_index(drop=True)
        present_files = {item.name for item in dataset_path.iterdir() if item.is_file()}
        if treasury_files.issubset(present_files):
            return load_treasury_yield_directory(dataset_path)
        if risk_files.issubset(present_files):
            return load_risk_sentiment_directory(dataset_path)
        if present_files.intersection(market_context_files):
            return load_market_context_directory(dataset_path)
        if risk_dir.is_dir():
            return load_risk_sentiment_directory(risk_dir)
        if market_context_dir.is_dir():
            return load_market_context_directory(market_context_dir)
        raise ValueError(
            "Directory must contain Treasury Yield and/or Risk Sentiment Metrics data files"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path, **read_csv_kwargs)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(dataset_path, sep="\t", **read_csv_kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path, **read_csv_kwargs)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def prepare_dataset(
    frame: pd.DataFrame,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    canonical = canonicalize_yield_frame(frame)
    enriched = add_required_metrics(canonical, window=rolling_window)
    enriched = add_market_context_metrics(enriched, window=rolling_window)
    enriched = add_nelson_siegel_factors(enriched)
    for factor in ("ns_level", "ns_slope", "ns_curvature"):
        enriched[f"z_{factor}"] = rolling_zscore(enriched[factor], rolling_window)
    return enriched


def _template_for_mapping(mapping: SignalTradeMapping) -> TradeTemplate:
    if mapping.trade_type.endswith("2s10s") or mapping.trade_type.startswith("2s10s_"):
        return TradeTemplate.CURVE_2S10S
    if mapping.trade_type.endswith("5s30s") or mapping.trade_type.startswith("5s30s_"):
        return TradeTemplate.CURVE_5S30S
    if "butterfly" in mapping.trade_type:
        return TradeTemplate.BUTTERFLY_2S5S10S
    return TradeTemplate.OUTRIGHT


def _legs_for_mapping(mapping: SignalTradeMapping) -> tuple[TradeLeg, ...]:
    normalized_weights = normalize_gross_weights(
        (leg.weight for leg in mapping.legs),
        target_gross=1.0,
    )
    return tuple(
        TradeLeg(
            tenor=leg.tenor,
            weight=weight if leg.side == "long" else -weight,
            duration=DURATION_ESTIMATES[leg.tenor],
        )
        for leg, weight in zip(mapping.legs, normalized_weights)
    )


def _pnl_for_legs(
    frame: pd.DataFrame,
    entry_index: int,
    exit_index: int,
    legs: tuple[TradeLeg, ...],
    gross_notional: float,
) -> float:
    pnl = 0.0
    for leg in legs:
        # FRED Treasury yields are expressed in percentage points. Convert a
        # move such as 4.00% -> 4.10% from 0.10 percentage points to 0.001.
        delta_yield_decimal = float(
            frame.iloc[exit_index][leg.tenor] - frame.iloc[entry_index][leg.tenor]
        ) / 100.0
        pnl += -gross_notional * leg.weight * leg.duration * delta_yield_decimal
    return pnl


def simulate_signal_events(
    frame: pd.DataFrame,
    events: Iterable[SignalEvent],
    config: BacktestConfig = BacktestConfig(),
) -> list[TradeResult]:
    results: list[TradeResult] = []
    index = frame.index
    for event in events:
        mapping = SIGNAL_TO_TRADE.get(event.signal_name)
        if mapping is None:
            continue
        try:
            signal_index = index.get_loc(event.date)
        except KeyError:
            continue
        if isinstance(signal_index, slice):
            signal_index = signal_index.start
        entry_index = int(signal_index) + config.signal_lag
        exit_index = entry_index + config.holding_period
        if entry_index >= len(index) or exit_index >= len(index):
            continue
        if config.gross_notional <= 0:
            raise ValueError("gross_notional must be positive")
        legs = _legs_for_mapping(mapping)
        pnl = _pnl_for_legs(
            frame,
            entry_index,
            exit_index,
            legs,
            config.gross_notional,
        )
        results.append(
            TradeResult(
                signal_name=event.signal_name,
                template=_template_for_mapping(mapping),
                entry_date=pd.Timestamp(index[entry_index]),
                exit_date=pd.Timestamp(index[exit_index]),
                pnl=float(pnl),
                gross_notional=float(config.gross_notional),
                legs=legs,
                metadata={
                    "signal_date": event.date.isoformat(),
                    "trade_type": mapping.trade_type,
                    "label": mapping.label,
                    "interpretation": event.interpretation,
                    "rationale": mapping.rationale,
                },
            )
        )
    return results


def run_backtest(
    frame: pd.DataFrame,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestRun:
    prepared = prepare_dataset(frame, rolling_window=rolling_window)
    signal_events = generate_signal_events(prepared)
    accepted_events, regime_decisions = apply_regime_filters(
        prepared,
        signal_events,
        enabled=config.use_regime_filters,
    )
    trades = simulate_signal_events(prepared, accepted_events, config=config)
    portfolio = summarize_portfolio(trades, annualization_factor=config.annualization_factor)
    return BacktestRun(
        frame=prepared,
        signal_events=signal_events,
        trades=trades,
        portfolio=portfolio,
        regime_decisions=regime_decisions,
    )


def run_backtest_from_path(
    path: str | Path,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestRun:
    frame = load_dataset(path)
    return run_backtest(
        frame,
        rolling_window=rolling_window,
        config=config,
    )


def backtest_run_to_dict(result: BacktestRun) -> dict[str, object]:
    breakdown = summarize_pnl_breakdown(result.trades)

    def breakdown_rows(rows: tuple[PnlBreakdownRow, ...]) -> list[dict[str, object]]:
        return [
            {
                "label": row.label,
                "trade_count": row.trade_count,
                "cumulative_pnl": row.cumulative_pnl,
                "average_pnl": row.average_pnl,
                "win_rate": row.win_rate,
            }
            for row in rows
        ]

    data_start = pd.Timestamp(result.frame.index.min()) if not result.frame.empty else None
    data_end = pd.Timestamp(result.frame.index.max()) if not result.frame.empty else None
    first_entry = min((trade.entry_date for trade in result.trades), default=None)
    last_exit = max((trade.exit_date for trade in result.trades), default=None)
    return {
        "period": {
            "data_start": data_start.isoformat() if data_start is not None else None,
            "data_end": data_end.isoformat() if data_end is not None else None,
            "first_trade_entry": first_entry.isoformat() if first_entry is not None else None,
            "last_trade_exit": last_exit.isoformat() if last_exit is not None else None,
        },
        "portfolio": {
            "cumulative_pnl": result.portfolio.cumulative_pnl,
            "sharpe_ratio": result.portfolio.sharpe_ratio,
            "max_drawdown": result.portfolio.max_drawdown,
            "win_rate": result.portfolio.win_rate,
            "trade_count": result.portfolio.trade_count,
        },
        "pnl_breakdown": {
            "average_winner": breakdown.average_winner,
            "average_loser": breakdown.average_loser,
            "profit_factor": breakdown.profit_factor,
            "by_category": breakdown_rows(breakdown.by_category),
            "by_signal": breakdown_rows(breakdown.by_signal),
            "by_exit_year": breakdown_rows(breakdown.by_exit_year),
        },
        "regime_filter": {
            "candidate_signals": len(result.signal_events),
            "accepted_signals": sum(decision.passed for decision in result.regime_decisions),
            "rejected_signals": sum(not decision.passed for decision in result.regime_decisions),
            "decisions": [
                {
                    "date": decision.event.date.isoformat(),
                    "signal_name": decision.event.signal_name,
                    "passed": decision.passed,
                    "filter_name": decision.filter_name,
                    "score": decision.score,
                    "reason": decision.reason,
                }
                for decision in result.regime_decisions
            ],
        },
        "signal_events": [
            {
                "date": event.date.isoformat(),
                "signal_name": event.signal_name,
                "interpretation": event.interpretation,
                "metric_name": event.metric_name,
                "metric_value": event.metric_value,
                "threshold": event.threshold,
            }
            for event in result.signal_events
        ],
        "trades": [
            {
                "signal_name": trade.signal_name,
                "template": trade.template.value,
                "entry_date": trade.entry_date.isoformat(),
                "exit_date": trade.exit_date.isoformat(),
                "pnl": trade.pnl,
                "gross_notional": trade.gross_notional,
                "legs": [
                    {
                        "tenor": leg.tenor,
                        "side": "long" if leg.weight > 0 else "short",
                        "weight": leg.weight,
                        "duration": leg.duration,
                    }
                    for leg in trade.legs
                ],
                "metadata": trade.metadata,
            }
            for trade in result.trades
        ],
    }
