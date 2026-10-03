from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .pipeline import BacktestRun, backtest_run_to_dict, run_backtest_from_path
from .signals import MEAN_REVERSION_SIGNALS, SIGNAL_RULES, SIGNAL_TO_TRADE
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
    "yield_selloff_exhaustion": "10Y Yield Selloff Reversal",
    "yield_rally_exhaustion": "10Y Yield Rally Reversal",
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
    starting_capital: float,
    use_regime_filters: bool,
    fresh_crossings_only: bool,
    one_active_trade_per_signal: bool,
    cooldown_period: int,
) -> BacktestRun:
    return run_backtest_from_path(
        Path(path),
        rolling_window=rolling_window,
        config=BacktestConfig(
            holding_period=holding_period,
            signal_lag=signal_lag,
            annualization_factor=annualization_factor,
            gross_notional=gross_notional,
            starting_capital=starting_capital,
            use_regime_filters=use_regime_filters,
            fresh_crossings_only=fresh_crossings_only,
            one_active_trade_per_signal=one_active_trade_per_signal,
            cooldown_period=cooldown_period,
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
                "Market Regime Filter": decision.filter_name if decision else "not recorded",
                "Regime score": decision.score if decision else None,
                "Acceptance reason": decision.reason if decision else "No filter decision recorded.",
                "Legs": legs,
            }
        )
    return pd.DataFrame(records)


def _equity_curve(result: BacktestRun) -> pd.DataFrame:
    if result.daily_portfolio.empty:
        return pd.DataFrame(columns=["Cumulative return"])
    return result.daily_portfolio[["cumulative_return"]].rename(
        columns={"cumulative_return": "Cumulative return"}
    )


def _equity_comparison(filtered_result: BacktestRun, unfiltered_result: BacktestRun) -> pd.DataFrame:
    comparison = pd.concat(
        [
            _equity_curve(unfiltered_result).rename(columns={"Cumulative return": "Without filters"}),
            _equity_curve(filtered_result).rename(columns={"Cumulative return": "With filters"}),
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
            ("Cumulative return", f"{unfiltered['cumulative_return']:.2%}", f"{filtered['cumulative_return']:.2%}", f"{filtered['cumulative_return'] - unfiltered['cumulative_return']:+.2%}"),
            ("Annualized Sharpe", f"{unfiltered['sharpe_ratio']:.2f}", f"{filtered['sharpe_ratio']:.2f}", f"{filtered['sharpe_ratio'] - unfiltered['sharpe_ratio']:+.2f}"),
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

    st.markdown(
        f"""
        <div class="period-strip">
            <span><b>Input data</b> {_date(period['data_start'])} → {_date(period['data_end'])}</span>
            <span><b>Completed trades</b> {_date(period['first_trade_entry'])} → {_date(period['last_trade_exit'])}</span>
            <span><b>Displayed mode</b> Market Regime Filter {'on' if use_regime_filters else 'off'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="headline_metrics"):
        columns = st.columns(6)
        columns[0].metric(
            "Cumulative P&L",
            _money(portfolio["cumulative_pnl"]),
            help="Sum of daily mark-to-market P&L across all simulated trades.",
        )
        columns[1].metric(
            "Cumulative return",
            f"{portfolio['cumulative_return']:.2%}",
            help="Ending portfolio value divided by starting portfolio capital, minus one.",
        )
        columns[2].metric(
            "Annualized Sharpe",
            f"{portfolio['sharpe_ratio']:.2f}",
            help="Mean daily portfolio return divided by daily return volatility, multiplied by the square root of 252.",
        )
        columns[3].metric(
            "Max drawdown",
            _money(portfolio["max_drawdown"]),
            help="Largest peak-to-trough decline in the daily marked-to-market portfolio value.",
        )
        columns[4].metric(
            "Win rate",
            f"{portfolio['win_rate']:.1%}",
            help="Percentage of completed trades whose approximate P&L is greater than zero.",
        )
        columns[5].metric(
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
    st.subheader("Market Regime Filter impact")
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
        st.subheader("Cumulative return")
        st.caption("Open trades are marked to market on every trading day using the duration approximation.")
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
    st.subheader("Market Regime Filter impact by strategy family")
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
    st.subheader("Market Regime Filter impact by signal")
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
                "Market Regime Filter": (
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
        events.groupby(["Signal", "Market Regime Filter"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Accepted", "Rejected"], fill_value=0)
        .sort_index()
    )
    accepted = int((events["Market Regime Filter"] == "Accepted").sum())
    rejected = int((events["Market Regime Filter"] == "Rejected").sum())
    st.caption(
        f"{len(events):,} candidate signals · {accepted:,} accepted · "
        f"{rejected:,} rejected by the Market Regime Filter"
    )
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
        visible = visible[visible["Market Regime Filter"].isin(statuses)]
        st.dataframe(visible.sort_values("Date", ascending=False), hide_index=True, width="stretch", height=420)


def _signal_dictionary(result: BacktestRun) -> pd.DataFrame:
    trigger_descriptions = {
        "hot_CPI": "CPI was significantly higher than the market expected",
        "cold_CPI": "CPI was significantly lower than the market expected",
        "strong_payrolls": "Employment growth was significantly stronger than the market expected",
        "weak_payrolls": "Employment growth was significantly weaker than the market expected",
        "yield_selloff_momentum": "The 10Y yield has risen unusually quickly over five days",
        "yield_rally_momentum": "The 10Y yield has fallen unusually quickly over five days",
        "yield_selloff_exhaustion": "The 10Y yield has risen unusually far over 20 days and may be overextended",
        "yield_rally_exhaustion": "The 10Y yield has fallen unusually far over 20 days and may be overextended",
        "curve_too_flat": "The 2s10s spread is unusually low, meaning the curve is very flat or inverted",
        "curve_too_steep": "The 2s10s spread is unusually high, meaning the curve is very steep",
        "long_end_too_flat": "The 5s30s spread is unusually low, meaning the long end is very flat",
        "long_end_too_steep": "The 5s30s spread is unusually high, meaning the long end is very steep",
        "five_y_cheap": "The 5Y yield is unusually high relative to the 2Y and 10Y yields, so the 5Y bond appears cheap",
        "five_y_rich": "The 5Y yield is unusually low relative to the 2Y and 10Y yields, so the 5Y bond appears rich",
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
        if rule.name in MEAN_REVERSION_SIGNALS:
            direction = "below" if rule.operator == ">" else "above"
            trigger = f"{rule.metric} crosses back {direction} {rule.threshold:+g}"
        else:
            direction = "above" if rule.operator == ">" else "below"
            trigger = f"{rule.metric} crosses {direction} {rule.threshold:+g}"
        weights = normalize_gross_weights(leg.weight for leg in mapping.legs)
        legs = ", ".join(
            f"{leg.side.title()} {leg.tenor} {weight:.1%}"
            for leg, weight in zip(mapping.legs, weights)
        )
        rows.append(
            {
                "Signal": _signal_display_name(rule.name),
                "Data availability": "Available" if rule.metric in result.frame.columns else "Missing",
                "Trigger": trigger,
                "What the trigger means": trigger_descriptions[rule.name],
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
    starting_capital: float,
    use_regime_filters: bool,
) -> None:
    st.subheader("Backtesting methodology")
    st.markdown(
        f"""
        The backtest follows four steps:

        1. Calculate a market feature.
        2. Convert the feature to a rolling z-score.
        3. Use the crossing direction that matches the trade idea.
        4. Enter after **{signal_lag} trading day(s)** and exit after **{holding_period} trading day(s)**.
        """
    )

    st.subheader("1. Features and z-scores")
    st.markdown(
        fr"""
        Every feature is compared with its previous **{rolling_window} observations**:

        $$z_t = \frac{{x_t - \mu_{{t-1}}}}{{\sigma_{{t-1}}}}$$

        - $x_t$ is the feature value today.
        - $\mu_{{t-1}}$ and $\sigma_{{t-1}}$ are the mean and standard deviation of the prior window.

        Today's value is excluded from its own historical benchmark, which prevents look-ahead bias.
        """
    )
    signal_features = pd.DataFrame(
        [
            ("10Y yield change — 5 days", "10Y(t) − 10Y(t−5)", "Momentum", "z_y10_change_5d"),
            ("10Y yield change — 20 days", "10Y(t) − 10Y(t−20)", "Reversal", "z_y10_change_20d"),
            ("2s10s spread", "10Y − 2Y", "Curve", "z_2s10s"),
            ("5s30s spread", "30Y − 5Y", "Curve", "z_5s30s"),
            ("2s5s10s butterfly", "5Y − (2Y + 10Y) ÷ 2", "Butterfly", "z_butterfly"),
            ("CPI surprise", "Actual CPI − expected CPI", "Macro surprise", "z_cpi_surprise"),
            ("Payroll surprise", "Actual payrolls − expected payrolls", "Macro surprise", "z_nfp_surprise"),
        ],
        columns=["Feature", "Formula", "Used for", "Model field"],
    )
    signal_features["Data status"] = signal_features["Model field"].map(
        lambda field: "Available" if field in result.frame.columns else "Missing"
    )
    st.dataframe(signal_features, hide_index=True, width="stretch")

    st.subheader("2. Signal and entry rules")
    st.markdown(
        f"""
        The approach of this backtester relies mainly on statistical deviations from recent history rather than a full
        fundamental valuation model. This is a limitation because a large z-score shows that a move is unusual,
        but does not prove that it will continue or reverse.

        - The rules are grouped into two approaches:
          - **Momentum** (use **±1 SD** threshold and trigger when the threshold is first crossed)
            - **Directional yield:** 5-day changes in the 10Y yield
          - **Mean Reversion** (use **±2 SD** threshold for more abnormal market deviation, and trigger when reading crosses back in)
            - **Directional yield:** 20-day yield reversal
            - **Curve:** 2s10s and 5s30s normalization
            - **Butterfly:** 5Y relative value against the 2Y and 10Y yields
        - The **{rolling_window}-day** lookback gives approximately one trading year of recent history.
        - The **{signal_lag}-day** entry lag prevents same-day execution on the signal. The common
        **{holding_period}-day** holding period provides a consistent short-term comparison across signals.

        These choices are simple and explainable, but they were not optimized or independently validated.
        """
    )
    st.dataframe(
        _signal_dictionary(result),
        hide_index=True,
        width="stretch",
        height=560,
        column_config={"Events": st.column_config.NumberColumn(format="%d")},
    )

    st.subheader("3. Market Regime Filter")
    st.markdown(
        r"""
        The Market Regime Filter can reject a candidate, but it cannot create a trade.

        $$\text{Yield-Direction Score} = \text{upward-yield votes} - \text{downward-yield votes}$$

        - A score of **+2 or higher** suggests rising yields, so a **long Treasury trade is rejected**.
        - A score of **−2 or lower** suggests falling yields, so a **short Treasury trade is rejected**.
        - Scores from **−1 to +1** do not reject a trade.
        """
    )
    if not use_regime_filters:
        st.warning("The Market Regime Filter is disabled; every candidate signal proceeds to execution when enough future data exist.")

    filter_structure = pd.DataFrame(
        [
            (
                "Market votes",
                "SPY, VIX, credit, dollar and inflation",
                "Whether the broader market favors yields rising or falling",
                "Long or short Treasury trades",
            ),
            (
                "Nelson–Siegel",
                "Treasury yield-curve level, slope and curvature",
                "Whether the level or shape of the Treasury curve supports the signal",
                "10Y, curve and butterfly trades",
            ),
        ],
        columns=["Check", "Inputs", "What it tells us", "Applied to"],
    )
    st.dataframe(filter_structure, hide_index=True, width="stretch")

    st.markdown(
        """
        We interpret market conditions from two different sources - external market indicators and the yield curve:
        - External market indicators show broader risk, credit and inflation conditions.
        - Treasury yield curve shows how interest-rate expectations differ across maturities.

        Market votes are used to assess long and short Treasury trades. For Nelson-Siegel, we use level for 10Y yield trades, slope for curve trades, and curvature for butterfly trades. A Nelson–Siegel check rejects a
        trade only when the relevant factor is at least one standard deviation from normal and points against the signal.
        """
    )

    st.markdown("**Detailed input rules**")
    regime_inputs = pd.DataFrame(
        [
            ("SPY", "5-day return ≥ +1% is risk-on; ≤ −1% is risk-off", "Directional Treasury trades", "spy_return_5d"),
            ("VIX", "5-day change ≤ −1 is risk-on; ≥ +1 is risk-off", "Directional Treasury trades", "vix_change_5d"),
            ("HY credit spread", "5-day tightening ≥ 5 bp is risk-on; widening ≥ 5 bp is risk-off", "Directional Treasury trades", "hy_oas_change_5d_bp"),
            ("IG credit spread", "5-day tightening ≥ 5 bp is risk-on; widening ≥ 5 bp is risk-off", "Directional Treasury trades", "ig_oas_change_5d_bp"),
            ("Broad dollar", "5-day return ≤ −0.5% is risk-on; ≥ +0.5% is risk-off", "Directional Treasury trades", "broad_dollar_return_5d"),
            ("Inflation breakevens", "Each 5-day move ≥ +0.05 points votes yields up; ≤ −0.05 votes yields down", "Directional Treasury trades", "inflation_regime_score"),
            ("Nelson–Siegel level", "An opposing z-score with magnitude ≥ 1 vetoes the yield-move signal", "Directional yield momentum and reversal", "z_ns_level"),
            ("Nelson–Siegel slope", "An opposing z-score with magnitude ≥ 1 vetoes the curve signal", "Curve trades", "z_ns_slope"),
            ("Nelson–Siegel curvature", "An opposing z-score with magnitude ≥ 1 vetoes the butterfly signal", "Butterfly trades", "z_ns_curvature"),
        ],
        columns=["Regime input", "Decision rule", "Applied to", "Model field"],
    )
    regime_inputs["Data status"] = regime_inputs["Model field"].map(
        lambda field: "Available" if field in result.frame.columns else "Missing"
    )
    st.dataframe(regime_inputs, hide_index=True, width="stretch")

    st.subheader("4. Position sizing and P&L")
    completed_trade_types = {
        str(trade.metadata.get("trade_type", "")) for trade in result.trades
    }
    positioning_summary = pd.DataFrame(
        [
            (
                "Outright 10Y",
                "1",
                "Long or short 10Y",
                "Full notional in one leg",
                "Movement in the 10Y yield",
                "Yes" if any("momentum" in trade_type or "reversal" in trade_type for trade_type in completed_trade_types) else "No",
            ),
            (
                "2s10s curve",
                "2",
                "2Y and 10Y in opposite directions",
                "Legs weighted using duration",
                "Change in the 2s10s spread",
                "Yes" if any(trade_type.startswith("2s10s_") for trade_type in completed_trade_types) else "No",
            ),
            (
                "5s30s curve",
                "2",
                "5Y and 30Y in opposite directions",
                "Legs weighted using duration",
                "Change in the 5s30s spread",
                "Yes" if any(trade_type.startswith("5s30s_") for trade_type in completed_trade_types) else "No",
            ),
            (
                "2s5s10s butterfly",
                "3",
                "5Y belly against the 2Y and 10Y wings",
                "Belly and wings weighted using duration",
                "5Y movement relative to the wings",
                "Yes" if any("butterfly" in trade_type for trade_type in completed_trade_types) else "No",
            ),
        ],
        columns=[
            "Trade structure",
            "Legs",
            "Position structure",
            "Sizing approach",
            "P&L driver",
            "Used in current results",
        ],
    )
    st.dataframe(positioning_summary, hide_index=True, width="stretch")

    st.subheader("5. Performance metrics")
    metrics = pd.DataFrame(
        [
            ("Starting Capital", _money(starting_capital)),
            ("Cumulative P&L", "Sum of daily mark-to-market P&L"),
            ("Cumulative return", "Ending portfolio value ÷ starting capital − 1"),
            (
                "Annualized Sharpe",
                f"mean(daily portfolio return) ÷ std(daily portfolio return) × √{annualization_factor}",
            ),
            ("Maximum drawdown", "Largest decline in daily portfolio value from a previous peak"),
            ("Win rate", "Winning trades ÷ completed trades"),
            ("Profit factor", "Total winning P&L ÷ |total losing P&L|"),
            ("Average winner / loser", "Mean P&L of winning trades / losing trades"),
            ("Trade count", "Completed trades after the overlap and cooldown rules"),
        ],
        columns=["Metric", "Formula or definition"],
    )
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.caption(
        "Open trades are revalued each trading day using duration and daily yield changes. Daily return is "
        "daily P&L divided by the previous portfolio value. The risk-free rate is set to zero; transaction "
        "costs, carry and convexity are excluded."
    )


def _render_trades(result: BacktestRun) -> None:
    trades = _trade_frame(result)
    if trades.empty:
        st.info("No trades completed for the selected configuration.")
        return
    st.subheader("Completed accepted-trade log")
    st.caption("Only candidates accepted by the selected Market Regime Filter mode can appear here.")
    signals = st.multiselect("Signal", sorted(trades["Signal"].unique()), key="trade_signals")
    outcomes = st.multiselect("Outcome", ["Win", "Loss", "Flat"], key="trade_outcomes")
    filter_types = st.multiselect(
        "Market Regime Filter type",
        sorted(trades["Market Regime Filter"].unique()),
        key="trade_filter_types",
    )
    visible = trades
    if signals:
        visible = visible[visible["Signal"].isin(signals)]
    if outcomes:
        visible = visible[visible["Result"].isin(outcomes)]
    if filter_types:
        visible = visible[visible["Market Regime Filter"].isin(filter_types)]
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

    st.subheader("Market Regime Filter diagnostics")
    st.caption(
        "A positive Yield-Direction Score favors rising yields; a negative score favors falling yields. "
        "Zero is neutral."
    )
    regime_columns = [
        column
        for column in ("risk_regime_score", "inflation_regime_score", "yield_direction_score")
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
        .st-key-headline_metrics [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; overflow-x: auto; padding-bottom: .55rem; }
        .st-key-headline_metrics [data-testid="column"] { min-width: 185px; flex: 0 0 185px; }
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
        starting_capital = st.number_input(
            "Starting Capital",
            min_value=100_000.0,
            max_value=1_000_000_000.0,
            value=10_000_000.0,
            step=1_000_000.0,
            format="%.0f",
            help="The value of the portfolio at the beginning of the backtest. It is used to convert daily P&L into daily returns.",
        )
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
        annualization_factor = 252
        cooldown_period = st.number_input(
            "Post-exit cooldown (trading days)",
            min_value=0,
            max_value=60,
            value=5,
            help=(
                "The number of trading days a signal must wait after its previous trade closes "
                "before it can open another trade. Other signals can still trade during this period."
            ),
        )
        fresh_crossings_only = st.checkbox(
            "New threshold crossings only",
            value=True,
            help=(
                "For momentum and macro signals, an existing threshold breach cannot trigger again. "
                "It must first move back inside the threshold and then cross outward again. "
                "Yield-reversal, curve and butterfly signals always wait for a crossing back inside."
            ),
        )
        one_active_trade_per_signal = st.checkbox(
            "One active trade per signal",
            value=True,
            help=(
                "The same signal cannot open a new trade while its previous trade is still active. "
                "Other signals can still open trades."
            ),
        )
        use_regime_filters = st.checkbox(
            "Use Market Regime Filter",
            value=True,
            help="Veto signals that conflict with available market context or Nelson–Siegel curve factors.",
        )
        if st.button("Refresh backtest", type="primary", width="stretch"):
            _run_cached.clear()
        st.divider()
        st.caption("Signals use lag-safe rolling statistics. New threshold crossings, active-position limits and cooldowns prevent repeated exposure.")

    try:
        with st.spinner("Running filtered and unfiltered Treasury strategies…"):
            filtered_result = _run_cached(
                data_path,
                int(rolling_window),
                int(holding_period),
                int(signal_lag),
                int(annualization_factor),
                float(gross_notional),
                float(starting_capital),
                True,
                bool(fresh_crossings_only),
                bool(one_active_trade_per_signal),
                int(cooldown_period),
            )
            unfiltered_result = _run_cached(
                data_path,
                int(rolling_window),
                int(holding_period),
                int(signal_lag),
                int(annualization_factor),
                float(gross_notional),
                float(starting_capital),
                False,
                bool(fresh_crossings_only),
                bool(one_active_trade_per_signal),
                int(cooldown_period),
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
            starting_capital=float(starting_capital),
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
