from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .pipeline import BacktestRun, backtest_run_to_dict, run_backtest_from_path
from .signals import SIGNAL_RULES, SIGNAL_TO_TRADE
from .trades import normalize_gross_weights
from .types import BacktestConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


SIGNAL_DISPLAY_NAMES = {
    "hot_CPI": "Hot CPI Surprise",
    "cold_CPI": "Cool CPI Surprise",
    "strong_payrolls": "Strong Payrolls Surprise",
    "weak_payrolls": "Weak Payrolls Surprise",
    "yield_selloff_momentum": "10Y Yield Selloff Momentum",
    "yield_rally_momentum": "10Y Yield Rally Momentum",
    "yield_selloff_exhaustion": "10Y Yield Selloff Exhaustion",
    "yield_rally_exhaustion": "10Y Yield Rally Exhaustion",
    "curve_too_flat": "2s10s Curve Too Flat",
    "curve_too_steep": "2s10s Curve Too Steep",
    "long_end_too_flat": "5s30s Curve Too Flat",
    "long_end_too_steep": "5s30s Curve Too Steep",
    "five_y_cheap": "5Y Belly Cheap",
    "five_y_rich": "5Y Belly Rich",
}


def _signal_display_name(name: str) -> str:
    return SIGNAL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def _run_cached(
    path: str,
    rolling_window: int,
    holding_period: int,
    signal_lag: int,
    annualization_factor: int,
    gross_notional: float,
    use_regime_filters: bool,
) -> BacktestRun:
    return run_backtest_from_path(
        Path(path),
        rolling_window=rolling_window,
        config=BacktestConfig(
            holding_period=holding_period,
            signal_lag=signal_lag,
            annualization_factor=annualization_factor,
            gross_notional=gross_notional,
            use_regime_filters=use_regime_filters,
        ),
    )


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _date(value: str | None) -> str:
    return value[:10] if value else "n/a"


def _breakdown_frame(rows: list[dict[str, object]], label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "label": label,
            "trade_count": "Trades",
            "cumulative_pnl": "P&L",
            "average_pnl": "Average P&L",
            "win_rate": "Win rate",
        }
    )
    if label == "Signal":
        frame[label] = frame[label].map(_signal_display_name)
    return frame


def _trade_frame(result: BacktestRun) -> pd.DataFrame:
    decisions = {
        (decision.event.date, decision.event.signal_name): decision
        for decision in result.regime_decisions
    }
    records = []
    for trade in result.trades:
        signal_date = pd.Timestamp(trade.metadata.get("signal_date", trade.entry_date))
        decision = decisions.get((signal_date, trade.signal_name))
        legs = ", ".join(
            f"{'Long' if leg.weight > 0 else 'Short'} {leg.tenor} ({abs(leg.weight):.1%})"
            for leg in trade.legs
        )
        records.append(
            {
                "Signal": _signal_display_name(trade.signal_name),
                "Strategy": trade.metadata.get("label", trade.template.value),
                "Signal date": signal_date,
                "Entry": trade.entry_date,
                "Exit": trade.exit_date,
                "P&L": trade.pnl,
                "Result": "Win" if trade.pnl > 0 else "Loss" if trade.pnl < 0 else "Flat",
                "Regime filter": decision.filter_name if decision else "not recorded",
                "Regime score": decision.score if decision else None,
                "Acceptance reason": decision.reason if decision else "No filter decision recorded.",
                "Legs": legs,
            }
        )
    return pd.DataFrame(records)


def _equity_curve(result: BacktestRun) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(columns=["Cumulative P&L"])
    realized = pd.DataFrame(
        {"date": [trade.exit_date for trade in result.trades], "pnl": [trade.pnl for trade in result.trades]}
    )
    daily = realized.groupby("date", as_index=True)["pnl"].sum().sort_index()
    return daily.cumsum().rename("Cumulative P&L").to_frame()


def _equity_comparison(filtered_result: BacktestRun, unfiltered_result: BacktestRun) -> pd.DataFrame:
    comparison = pd.concat(
        [
            _equity_curve(unfiltered_result).rename(columns={"Cumulative P&L": "Without filters"}),
            _equity_curve(filtered_result).rename(columns={"Cumulative P&L": "With filters"}),
        ],
        axis=1,
    ).sort_index()
    return comparison.ffill().fillna(0.0)


def _performance_comparison(
    filtered_payload: dict[str, object],
    unfiltered_payload: dict[str, object],
) -> pd.DataFrame:
    filtered = filtered_payload["portfolio"]
    unfiltered = unfiltered_payload["portfolio"]
    filtered_breakdown = filtered_payload["pnl_breakdown"]
    unfiltered_breakdown = unfiltered_payload["pnl_breakdown"]

    def ratio(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    return pd.DataFrame(
        [
            ("Cumulative P&L", _money(unfiltered["cumulative_pnl"]), _money(filtered["cumulative_pnl"]), _money(filtered["cumulative_pnl"] - unfiltered["cumulative_pnl"])),
            ("Sharpe ratio", f"{unfiltered['sharpe_ratio']:.2f}", f"{filtered['sharpe_ratio']:.2f}", f"{filtered['sharpe_ratio'] - unfiltered['sharpe_ratio']:+.2f}"),
            ("Maximum drawdown", _money(unfiltered["max_drawdown"]), _money(filtered["max_drawdown"]), _money(filtered["max_drawdown"] - unfiltered["max_drawdown"])),
            ("Win rate", f"{unfiltered['win_rate']:.1%}", f"{filtered['win_rate']:.1%}", f"{filtered['win_rate'] - unfiltered['win_rate']:+.1%}"),
            ("Profit factor", ratio(unfiltered_breakdown["profit_factor"]), ratio(filtered_breakdown["profit_factor"]), "—"),
            ("Completed trades", f"{unfiltered['trade_count']:,}", f"{filtered['trade_count']:,}", f"{filtered['trade_count'] - unfiltered['trade_count']:+,}"),
        ],
        columns=["Metric", "Without filters", "With filters", "Change"],
    )


def _breakdown_comparison(
    filtered_payload: dict[str, object],
    unfiltered_payload: dict[str, object],
    key: str,
    label: str,
) -> pd.DataFrame:
    filtered = _breakdown_frame(filtered_payload["pnl_breakdown"][key], label)
    unfiltered = _breakdown_frame(unfiltered_payload["pnl_breakdown"][key], label)
    filtered = filtered[[label, "Trades", "P&L"]].rename(
        columns={"Trades": "Trades with filters", "P&L": "P&L with filters"}
    )
    unfiltered = unfiltered[[label, "Trades", "P&L"]].rename(
        columns={"Trades": "Trades without filters", "P&L": "P&L without filters"}
    )
    comparison = unfiltered.merge(filtered, on=label, how="outer").fillna(0)
    comparison["Trades removed"] = comparison["Trades without filters"] - comparison["Trades with filters"]
    comparison["P&L change"] = comparison["P&L with filters"] - comparison["P&L without filters"]
    return comparison.sort_values("P&L change")


def _render_breakdown_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No completed trades are available for this breakdown.")
        return
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Average P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Win rate": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.1%%"),
        },
    )


def _render_header(payload: dict[str, object], use_regime_filters: bool) -> None:
    period = payload["period"]
    portfolio = payload["portfolio"]
    breakdown = payload["pnl_breakdown"]

    st.markdown('<p class="eyebrow">SYSTEMATIC RATES RESEARCH</p>', unsafe_allow_html=True)
    st.title("Treasury Rates Backtester")
    st.caption("Duration, curve and butterfly signals translated into reproducible five-day trade simulations.")

    st.markdown(
        f"""
        <div class="period-strip">
            <span><b>Input data</b> {_date(period['data_start'])} → {_date(period['data_end'])}</span>
            <span><b>Completed trades</b> {_date(period['first_trade_entry'])} → {_date(period['last_trade_exit'])}</span>
            <span><b>Displayed mode</b> Regime filters {'on' if use_regime_filters else 'off'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)
    columns[0].metric(
        "Cumulative P&L",
        _money(portfolio["cumulative_pnl"]),
        help="Sum of approximate P&L from all completed trades. P&L is recognized on exit dates.",
    )
    columns[1].metric(
        "Sharpe ratio",
        f"{portfolio['sharpe_ratio']:.2f}",
        help="Mean daily realized P&L divided by its population standard deviation, annualized by the selected factor.",
    )
    columns[2].metric(
        "Max drawdown",
        _money(portfolio["max_drawdown"]),
        help="Largest peak-to-trough decline in cumulative realized P&L, including a zero starting baseline.",
    )
    columns[3].metric(
        "Win rate",
        f"{portfolio['win_rate']:.1%}",
        help="Percentage of completed trades whose approximate P&L is greater than zero.",
    )
    columns[4].metric(
        "Profit factor",
        f"{breakdown['profit_factor']:.2f}" if breakdown["profit_factor"] else "n/a",
        help="Gross profit from winning trades divided by the absolute gross loss from losing trades.",
    )


def _render_overview(
    result: BacktestRun,
    payload: dict[str, object],
    filtered_result: BacktestRun,
    filtered_payload: dict[str, object],
    unfiltered_result: BacktestRun,
    unfiltered_payload: dict[str, object],
) -> None:
    breakdown = payload["pnl_breakdown"]
    st.subheader("Regime-filter impact")
    candidates = len(filtered_result.regime_decisions)
    accepted = sum(decision.passed for decision in filtered_result.regime_decisions)
    rejected = candidates - accepted
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate signals", f"{candidates:,}")
    c2.metric("Accepted signals", f"{accepted:,}")
    c3.metric("Rejected signals", f"{rejected:,}")
    c4.metric(
        "Completed filtered trades",
        f"{filtered_result.portfolio.trade_count:,}",
        help="Accepted signals with enough remaining data to enter and complete the configured holding period.",
    )
    st.dataframe(
        _performance_comparison(filtered_payload, unfiltered_payload),
        hide_index=True,
        width="stretch",
    )

    left, right = st.columns((1.65, 1.0), gap="large")
    with left:
        st.subheader("Realized equity curves")
        st.caption("P&L is recognized on each trade's exit date; this is not daily mark-to-market accounting.")
        st.line_chart(_equity_comparison(filtered_result, unfiltered_result), height=345)
    with right:
        st.subheader("P&L by strategy family")
        category = _breakdown_frame(breakdown["by_category"], "Category")
        if not category.empty:
            chart = category.set_index("Category")[["P&L"]]
            st.bar_chart(chart, color="#f2a93b", horizontal=True, height=345)

    st.subheader("Annual realized P&L")
    yearly = _breakdown_frame(breakdown["by_exit_year"], "Exit year")
    if not yearly.empty:
        st.bar_chart(yearly.set_index("Exit year")[["P&L"]], color="#4dabf7", height=280)

    winner, loser, activity = st.columns(3)
    winner.metric("Average winner", _money(breakdown["average_winner"]))
    loser.metric("Average loser", _money(breakdown["average_loser"]))
    activity.metric("Completed trades", f"{payload['portfolio']['trade_count']:,}")


def _render_breakdowns(
    payload: dict[str, object],
    filtered_payload: dict[str, object],
    unfiltered_payload: dict[str, object],
) -> None:
    breakdown = payload["pnl_breakdown"]
    st.subheader("Regime-filter impact by strategy family")
    st.dataframe(
        _breakdown_comparison(filtered_payload, unfiltered_payload, "by_category", "Category"),
        hide_index=True,
        width="stretch",
        column_config={
            "P&L without filters": st.column_config.NumberColumn(format="$%.2f"),
            "P&L with filters": st.column_config.NumberColumn(format="$%.2f"),
            "P&L change": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.subheader("Regime-filter impact by signal")
    st.dataframe(
        _breakdown_comparison(filtered_payload, unfiltered_payload, "by_signal", "Signal"),
        hide_index=True,
        width="stretch",
        column_config={
            "P&L without filters": st.column_config.NumberColumn(format="$%.2f"),
            "P&L with filters": st.column_config.NumberColumn(format="$%.2f"),
            "P&L change": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.subheader("Strategy-family performance")
    _render_breakdown_table(_breakdown_frame(breakdown["by_category"], "Category"))
    st.subheader("Individual signal performance")
    _render_breakdown_table(_breakdown_frame(breakdown["by_signal"], "Signal"))
    st.subheader("Performance by exit year")
    _render_breakdown_table(_breakdown_frame(breakdown["by_exit_year"], "Exit year"))


def _render_signals(result: BacktestRun) -> None:
    decisions = {
        (decision.event.date, decision.event.signal_name): decision
        for decision in result.regime_decisions
    }
    events = pd.DataFrame(
        [
            {
                "Date": event.date,
                "Signal": _signal_display_name(event.signal_name),
                "Metric": event.metric_name,
                "Value": event.metric_value,
                "Threshold": event.threshold,
                "Interpretation": event.interpretation,
                "Regime filter": (
                    "Accepted"
                    if decisions.get((event.date, event.signal_name), None) is None
                    or decisions[(event.date, event.signal_name)].passed
                    else "Rejected"
                ),
                "Filter type": (
                    decisions[(event.date, event.signal_name)].filter_name
                    if (event.date, event.signal_name) in decisions
                    else "not recorded"
                ),
                "Regime score": (
                    decisions[(event.date, event.signal_name)].score
                    if (event.date, event.signal_name) in decisions
                    else None
                ),
                "Filter reason": (
                    decisions[(event.date, event.signal_name)].reason
                    if (event.date, event.signal_name) in decisions
                    else "No filter decision recorded."
                ),
            }
            for event in result.signal_events
        ]
    )
    if events.empty:
        st.info("No signals fired for the selected configuration.")
        return
    counts = (
        events.groupby(["Signal", "Regime filter"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Accepted", "Rejected"], fill_value=0)
        .sort_index()
    )
    accepted = int((events["Regime filter"] == "Accepted").sum())
    rejected = int((events["Regime filter"] == "Rejected").sum())
    st.caption(f"{len(events):,} candidate signals · {accepted:,} accepted · {rejected:,} rejected by regime filters")
    left, right = st.columns((1.0, 1.6), gap="large")
    with left:
        st.subheader("Candidate-signal outcomes")
        st.bar_chart(counts, color=["#20c997", "#ff6b6b"], horizontal=True, height=420)
    with right:
        st.subheader("Signal-event log")
        selected = st.multiselect("Filter signals", sorted(events["Signal"].unique()))
        statuses = st.multiselect(
            "Filter outcomes",
            ["Accepted", "Rejected"],
            default=["Accepted", "Rejected"],
        )
        visible = events[events["Signal"].isin(selected)] if selected else events
        visible = visible[visible["Regime filter"].isin(statuses)]
        st.dataframe(visible.sort_values("Date", ascending=False), hide_index=True, width="stretch", height=420)


def _signal_dictionary(result: BacktestRun) -> pd.DataFrame:
    feature_descriptions = {
        "z_cpi_surprise": "CPI actual minus consensus, standardized against prior observations",
        "z_nfp_surprise": "Payroll actual minus consensus, standardized against prior observations",
        "z_y10_change_5d": "5-day change in the 10Y yield, standardized against prior history",
        "z_y10_change_20d": "20-day change in the 10Y yield, standardized against prior history",
        "z_2s10s": "10Y minus 2Y yield spread, standardized against prior history",
        "z_5s30s": "30Y minus 5Y yield spread, standardized against prior history",
        "z_butterfly": "5Y minus the average of 2Y and 10Y yields, standardized against prior history",
    }
    event_counts: dict[str, int] = {}
    accepted_counts: dict[str, int] = {}
    for event in result.signal_events:
        event_counts[event.signal_name] = event_counts.get(event.signal_name, 0) + 1
    for decision in result.regime_decisions:
        if decision.passed:
            accepted_counts[decision.event.signal_name] = accepted_counts.get(decision.event.signal_name, 0) + 1

    rows = []
    for rule in SIGNAL_RULES:
        mapping = SIGNAL_TO_TRADE[rule.name]
        weights = normalize_gross_weights(leg.weight for leg in mapping.legs)
        legs = ", ".join(
            f"{leg.side.title()} {leg.tenor} {weight:.1%}"
            for leg, weight in zip(mapping.legs, weights)
        )
        rows.append(
            {
                "Signal": _signal_display_name(rule.name),
                "Input status": "Available" if rule.metric in result.frame.columns else "Missing",
                "Trigger": f"{rule.metric} {rule.operator} {rule.threshold:+g}",
                "Feature meaning": feature_descriptions[rule.metric],
                "Trade": mapping.label,
                "Gross allocation": legs,
                "Events": event_counts.get(rule.name, 0),
                "Accepted": accepted_counts.get(rule.name, 0),
                "Economic idea": rule.interpretation,
            }
        )
    return pd.DataFrame(rows)


def _render_methodology(
    result: BacktestRun,
    *,
    rolling_window: int,
    holding_period: int,
    signal_lag: int,
    annualization_factor: int,
    use_regime_filters: bool,
) -> None:
    st.subheader("Backtesting Methodology Used")
    st.markdown(
        f"""
        Each signal input is converted to a **rolling z-score**: today's value minus the mean of the prior
        **{rolling_window} observations**, divided by their standard deviation. The rolling reference
        window is shifted by one row, so today's observation is never included in its own benchmark.
        When a rule crosses its threshold, the engine enters after **{signal_lag} trading day(s)**,
        holds for **{holding_period} trading day(s)**, and exits. A rule that remains true on consecutive
        dates launches a separate, potentially overlapping trade on each date.
        """
    )

    st.subheader("Signal input features")
    st.caption("These variables generate candidate trades after being standardized into rolling z-scores.")
    signal_features = pd.DataFrame(
        [
            ("10Y yield change — 5 days", "10Y yield today minus its level five trading rows earlier", "Momentum", "z_y10_change_5d"),
            ("10Y yield change — 20 days", "10Y yield today minus its level 20 trading rows earlier", "Exhaustion / reversal", "z_y10_change_20d"),
            ("2s10s spread", "10Y yield minus 2Y yield", "Curve steepener / flattener", "z_2s10s"),
            ("5s30s spread", "30Y yield minus 5Y yield", "Long-end steepener / flattener", "z_5s30s"),
            ("2s5s10s butterfly", "5Y yield minus the average of the 2Y and 10Y yields", "5Y relative-value butterfly", "z_butterfly"),
            ("CPI surprise", "Released CPI actual minus the pre-release consensus forecast", "Directional duration", "z_cpi_surprise"),
            ("Payroll surprise", "Released nonfarm-payroll actual minus the pre-release consensus forecast", "Directional duration", "z_nfp_surprise"),
        ],
        columns=["Feature", "Raw definition", "Used for", "Model field"],
    )
    signal_features["Data status"] = signal_features["Model field"].map(
        lambda field: "Available" if field in result.frame.columns else "Missing"
    )
    st.dataframe(signal_features, hide_index=True, width="stretch")

    st.subheader("Regime filters")
    st.markdown(
        """
        A **regime filter is a second-stage check on a candidate trade**. The signal feature and threshold
        decide whether a trade idea exists; the regime filter then asks whether the broader market backdrop
        strongly contradicts that idea. It never creates a trade by itself.

        The process is: **signal threshold fires → candidate trade is created → regime evidence is checked →
        accepted candidates are executed after the entry lag**. This implementation is a conflict veto:
        neutral or unavailable regime data allows the candidate through, while clearly opposing evidence
        rejects it.
        """
    )
    if not use_regime_filters:
        st.warning("Regime filtering is disabled; every candidate signal proceeds to execution when enough future data exist.")

    regime_inputs = pd.DataFrame(
        [
            ("SPY", "5-day return ≥ +1% is risk-on; ≤ −1% is risk-off", "Directional duration", "spy_return_5d"),
            ("VIX", "5-day change ≤ −1 is risk-on; ≥ +1 is risk-off", "Directional duration", "vix_change_5d"),
            ("HY credit spread", "5-day tightening ≥ 5 bp is risk-on; widening ≥ 5 bp is risk-off", "Directional duration", "hy_oas_change_5d_bp"),
            ("IG credit spread", "5-day tightening ≥ 5 bp is risk-on; widening ≥ 5 bp is risk-off", "Directional duration", "ig_oas_change_5d_bp"),
            ("Broad dollar", "5-day return ≤ −0.5% is risk-on; ≥ +0.5% is risk-off", "Directional duration", "broad_dollar_return_5d"),
            ("Inflation breakevens", "Each 5-day move ≥ +0.05 points votes yields up; ≤ −0.05 votes yields down", "Directional duration", "inflation_regime_score"),
            ("Nelson–Siegel level", "An opposing z-score with magnitude ≥ 1 vetoes the yield-move signal", "Momentum and exhaustion", "z_ns_level"),
            ("Nelson–Siegel slope", "An opposing z-score with magnitude ≥ 1 vetoes the curve signal", "Curve trades", "z_ns_slope"),
            ("Nelson–Siegel curvature", "An opposing z-score with magnitude ≥ 1 vetoes the butterfly signal", "Butterfly trades", "z_ns_curvature"),
        ],
        columns=["Regime input", "Decision rule", "Applied to", "Model field"],
    )
    regime_inputs["Data status"] = regime_inputs["Model field"].map(
        lambda field: "Available" if field in result.frame.columns else "Missing"
    )
    st.dataframe(regime_inputs, hide_index=True, width="stretch")

    st.subheader("Signal rules and resulting trades")
    st.caption(
        "Available means the required feature exists in the loaded dataset. Missing macro surprise inputs cannot fire. "
        "Events counts every date on which the rule is true."
    )
    st.dataframe(
        _signal_dictionary(result),
        hide_index=True,
        width="stretch",
        height=560,
        column_config={"Events": st.column_config.NumberColumn(format="%d")},
    )

    st.subheader("Position sizing and P&L")
    st.markdown(
        """
        The selected gross notional is divided among a trade's legs so the absolute weights sum to 100%.
        Curve and butterfly templates use duration estimates to offset first-order duration exposure across
        their legs. Outright positions and the equal-notional 2Y/10Y macro trades are directional and are
        **not generally duration-neutral**.

        For each leg, approximate P&L is `−notional × signed weight × duration × yield change`, with the
        yield change converted from percentage points to decimals. Long bonds therefore profit when yields
        fall; short bonds profit when yields rise. The calculation excludes carry, roll-down, convexity,
        financing, bid/ask spreads, fees, and slippage.
        """
    )

    st.subheader("Performance metrics")
    metrics = pd.DataFrame(
        [
            ("Cumulative P&L", "Sum of approximate dollar P&L across completed trades."),
            (
                "Sharpe ratio",
                f"Mean daily realized P&L ÷ population standard deviation of daily realized P&L × √{annualization_factor}. Days are grouped by trade exit date; this is not a return-based or mark-to-market Sharpe.",
            ),
            ("Maximum drawdown", "Largest peak-to-trough fall in cumulative realized P&L, measured from a zero initial baseline."),
            ("Win rate", "Completed winning trades ÷ all completed trades. A flat trade is not a win."),
            ("Profit factor", "Total P&L on winners ÷ absolute total P&L on losers. Above 1 means gross winners exceed gross losers."),
            ("Average winner / loser", "Mean P&L conditional on a trade finishing positive / negative."),
            ("Trade count", "Number of completed signal instances, not the number of unique signal types or non-overlapping positions."),
        ],
        columns=["Metric", "Definition"],
    )
    st.dataframe(metrics, hide_index=True, width="stretch")


def _render_trades(result: BacktestRun) -> None:
    trades = _trade_frame(result)
    if trades.empty:
        st.info("No trades completed for the selected configuration.")
        return
    st.subheader("Completed accepted-trade log")
    st.caption("Only candidates accepted by the selected regime-filter mode can appear here.")
    signals = st.multiselect("Signal", sorted(trades["Signal"].unique()), key="trade_signals")
    outcomes = st.multiselect("Outcome", ["Win", "Loss", "Flat"], key="trade_outcomes")
    filter_types = st.multiselect(
        "Regime filter type",
        sorted(trades["Regime filter"].unique()),
        key="trade_filter_types",
    )
    visible = trades
    if signals:
        visible = visible[visible["Signal"].isin(signals)]
    if outcomes:
        visible = visible[visible["Result"].isin(outcomes)]
    if filter_types:
        visible = visible[visible["Regime filter"].isin(filter_types)]
    st.caption(f"Showing {len(visible):,} of {len(trades):,} completed trades")
    st.dataframe(
        visible.sort_values("Entry", ascending=False),
        hide_index=True,
        width="stretch",
        height=600,
        column_config={"P&L": st.column_config.NumberColumn(format="$%.2f")},
    )


def _render_curve(result: BacktestRun) -> None:
    st.subheader("Treasury yield history")
    yields = result.frame[[column for column in ("2Y", "5Y", "10Y", "30Y") if column in result.frame]]
    st.line_chart(yields, height=340)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Curve spreads")
        spreads = result.frame[[column for column in ("spread_2s10s", "spread_5s30s", "fly_2s5s10s") if column in result.frame]]
        st.line_chart(spreads, height=320)
    with right:
        st.subheader("Nelson–Siegel factors")
        factors = result.frame[[column for column in ("ns_level", "ns_slope", "ns_curvature") if column in result.frame]]
        st.line_chart(factors, height=320)

    st.subheader("Regime-filter diagnostics")
    st.caption(
        "Positive duration-regime values favor rising yields and short-duration trades; negative values favor "
        "falling yields and long-duration trades. Zero is neutral."
    )
    regime_columns = [
        column
        for column in ("risk_regime_score", "inflation_regime_score", "duration_regime_score")
        if column in result.frame
    ]
    shape_columns = [
        column
        for column in ("z_ns_level", "z_ns_slope", "z_ns_curvature")
        if column in result.frame
    ]
    regime_left, regime_right = st.columns(2, gap="large")
    with regime_left:
        st.markdown("**Market-context scores**")
        if regime_columns:
            st.line_chart(result.frame[regime_columns].astype(float), height=300)
        else:
            st.info("No market-context score is available in the selected dataset.")
    with regime_right:
        st.markdown("**Nelson–Siegel filter z-scores**")
        if shape_columns:
            st.line_chart(result.frame[shape_columns], height=300)
        else:
            st.info("No Nelson–Siegel filter factors are available in the selected dataset.")


def _page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #08111f; }
        [data-testid="stSidebar"] { background: #0d1929; border-right: 1px solid #20314a; }
        .block-container { max-width: 1420px; padding-top: 2.25rem; padding-bottom: 4rem; }
        .eyebrow { color: #20c997; letter-spacing: .16em; font-size: .76rem; font-weight: 700; margin-bottom: -.5rem; }
        .period-strip { display: flex; flex-wrap: wrap; gap: 1rem 2rem; padding: .85rem 1rem; margin: 1.2rem 0 1.4rem; background: #0d1929; border: 1px solid #20314a; border-radius: 10px; color: #b8c7da; }
        .period-strip b { color: #f4f7fb; margin-right: .45rem; }
        [data-testid="stMetric"] { background: #0d1929; border: 1px solid #20314a; padding: 1rem; border-radius: 10px; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        h1, h2, h3 { letter-spacing: -.025em; }
        div[data-baseweb="tab-list"] { gap: .4rem; }
        button[data-baseweb="tab"] { background: #0d1929; border-radius: 8px; padding: .55rem .9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Treasury Rates Backtester", page_icon="📈", layout="wide")
    _page_style()

    with st.sidebar:
        st.markdown("## Backtest controls")
        data_path = str(REPOSITORY_ROOT)
        st.caption("Dataset: bundled Treasury Yield and Risk Sentiment CSV files")
        rolling_window = st.number_input("Rolling lookback", min_value=20, max_value=1000, value=252, step=21)
        holding_period = st.number_input("Holding period (trading days)", min_value=1, max_value=60, value=5)
        signal_lag = st.number_input("Entry lag (trading days)", min_value=0, max_value=10, value=1)
        gross_notional = st.number_input(
            "Gross notional per trade",
            min_value=10_000.0,
            max_value=100_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
            format="%.0f",
        )
        annualization_factor = st.number_input("Sharpe annualization", min_value=1, max_value=365, value=252)
        use_regime_filters = st.checkbox(
            "Use regime filters",
            value=True,
            help="Veto signals that conflict with available market context or Nelson–Siegel curve factors.",
        )
        if st.button("Refresh backtest", type="primary", width="stretch"):
            _run_cached.clear()
        st.divider()
        st.caption("Signals use historical rolling statistics and enter after the configured lag. Repeated signals can create overlapping positions. Regime filters veto conflicts but do not create trades.")

    try:
        with st.spinner("Running filtered and unfiltered Treasury strategies…"):
            filtered_result = _run_cached(
                data_path,
                int(rolling_window),
                int(holding_period),
                int(signal_lag),
                int(annualization_factor),
                float(gross_notional),
                True,
            )
            unfiltered_result = _run_cached(
                data_path,
                int(rolling_window),
                int(holding_period),
                int(signal_lag),
                int(annualization_factor),
                float(gross_notional),
                False,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Backtest could not run: {exc}")
        st.stop()

    result = filtered_result if use_regime_filters else unfiltered_result
    payload = backtest_run_to_dict(result)
    filtered_payload = backtest_run_to_dict(filtered_result)
    unfiltered_payload = backtest_run_to_dict(unfiltered_result)
    _render_header(payload, bool(use_regime_filters))

    methodology, overview, breakdowns, signals, trades, curve = st.tabs(
        ["Methodology", "Overview", "P&L breakdown", "Signals", "Trade log", "Yield Curve Analytics"]
    )
    with methodology:
        _render_methodology(
            result,
            rolling_window=int(rolling_window),
            holding_period=int(holding_period),
            signal_lag=int(signal_lag),
            annualization_factor=int(annualization_factor),
            use_regime_filters=bool(use_regime_filters),
        )
    with overview:
        _render_overview(
            result,
            payload,
            filtered_result,
            filtered_payload,
            unfiltered_result,
            unfiltered_payload,
        )
    with breakdowns:
        _render_breakdowns(payload, filtered_payload, unfiltered_payload)
    with signals:
        _render_signals(result)
    with trades:
        _render_trades(result)
    with curve:
        _render_curve(result)

    st.divider()
    st.caption("Research output only. P&L uses a first-order duration approximation and excludes carry, convexity, financing and execution costs.")


if __name__ == "__main__":
    main()
