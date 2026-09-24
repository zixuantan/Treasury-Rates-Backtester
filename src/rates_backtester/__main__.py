from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import BacktestRun, backtest_run_to_dict, run_backtest_from_path
from .types import BacktestConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Treasury rates backtester on a dataset folder or file.")
    parser.add_argument("path", nargs="?", default=".", help="Dataset file or folder. Defaults to the workspace root.")
    parser.add_argument("--rolling-window", type=int, default=252, help="Rolling window used for z-scores.")
    parser.add_argument("--holding-period", type=int, default=5, help="Trade holding period in trading days.")
    parser.add_argument("--signal-lag", type=int, default=1, help="Trading delay after a signal fires.")
    parser.add_argument("--annualization-factor", type=int, default=252, help="Annualization factor for Sharpe ratio.")
    parser.add_argument("--gross-notional", type=float, default=1_000_000.0, help="Gross dollar notional used for simulated trades.")
    parser.add_argument("--no-regime-filters", action="store_true", help="Disable market-context and Nelson–Siegel conflict filters.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    return parser


def _print_summary(result: BacktestRun) -> None:
    payload = backtest_run_to_dict(result)
    period = payload["period"]
    portfolio = payload["portfolio"]
    breakdown = payload["pnl_breakdown"]

    def date_only(value: str | None) -> str:
        return value[:10] if value else "n/a"

    def number(value: float | None, format_spec: str) -> str:
        return format(value, format_spec) if value is not None else "n/a"

    def print_breakdown(title: str, rows: list[dict[str, object]]) -> None:
        print(f"\n{title}")
        print(f"  {'name':<28} {'trades':>7} {'pnl':>14} {'avg pnl':>14} {'win rate':>10}")
        for row in rows:
            print(
                f"  {str(row['label']):<28} "
                f"{int(row['trade_count']):>7d} "
                f"{float(row['cumulative_pnl']):>14,.2f} "
                f"{float(row['average_pnl']):>14,.2f} "
                f"{float(row['win_rate']):>9.2%}"
            )

    print("Backtest period")
    print(f"  data: {date_only(period['data_start'])} to {date_only(period['data_end'])}")
    print(
        "  completed trades: "
        f"{date_only(period['first_trade_entry'])} to {date_only(period['last_trade_exit'])}"
    )

    print("Backtest summary")
    print(f"  cumulative_pnl: {portfolio['cumulative_pnl']:,.2f}")
    print(f"  sharpe_ratio: {portfolio['sharpe_ratio']:.4f}")
    print(f"  max_drawdown: {portfolio['max_drawdown']:,.2f}")
    print(f"  win_rate: {portfolio['win_rate']:.2%}")
    print(f"  average_winner: {number(breakdown['average_winner'], ',.2f')}")
    print(f"  average_loser: {number(breakdown['average_loser'], ',.2f')}")
    print(f"  profit_factor: {number(breakdown['profit_factor'], '.4f')}")
    print(f"  trade_count: {portfolio['trade_count']}")
    print(f"  signal_events: {len(result.signal_events)}")
    if result.regime_decisions:
        print(f"  regime_filter_accepted: {sum(decision.passed for decision in result.regime_decisions)}")
        print(f"  regime_filter_rejected: {sum(not decision.passed for decision in result.regime_decisions)}")

    print_breakdown("P&L by strategy category", breakdown["by_category"])
    print_breakdown("P&L by signal", breakdown["by_signal"])
    print_breakdown("P&L by exit year", breakdown["by_exit_year"])


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = BacktestConfig(
        holding_period=args.holding_period,
        signal_lag=args.signal_lag,
        annualization_factor=args.annualization_factor,
        gross_notional=args.gross_notional,
        use_regime_filters=not args.no_regime_filters,
    )
    result = run_backtest_from_path(
        Path(args.path),
        rolling_window=args.rolling_window,
        config=config,
    )
    if args.json:
        print(json.dumps(backtest_run_to_dict(result), indent=2))
        return
    _print_summary(result)


if __name__ == "__main__":
    main()
