from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .trades import (
    SignalTradeMapping,
    build_butterfly_mapping,
    build_curve_mapping,
    build_multi_leg_mapping,
    build_outright_mapping,
)


@dataclass(frozen=True)
class SignalRule:
    name: str
    metric: str
    operator: str
    threshold: float
    interpretation: str


@dataclass(frozen=True)
class SignalEvent:
    date: pd.Timestamp
    signal_name: str
    interpretation: str
    metric_name: str
    metric_value: float
    threshold: float


SIGNAL_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        name="hot_CPI",
        metric="z_cpi_surprise",
        operator=">",
        threshold=1.0,
        interpretation="Inflation came in materially hotter than expected. Market may price a more hawkish Fed path, pushing yields higher.",
    ),
    SignalRule(
        name="cold_CPI",
        metric="z_cpi_surprise",
        operator="<",
        threshold=-1.0,
        interpretation="Inflation came in materially cooler than expected. Market may price more cuts / lower policy rates, pushing yields lower.",
    ),
    SignalRule(
        name="strong_payrolls",
        metric="z_nfp_surprise",
        operator=">",
        threshold=1.0,
        interpretation="Labour market is stronger than expected. Growth is resilient; market may price fewer cuts or higher-for-longer rates.",
    ),
    SignalRule(
        name="weak_payrolls",
        metric="z_nfp_surprise",
        operator="<",
        threshold=-1.0,
        interpretation="Labour market is weaker than expected. Market may price more cuts and lower front-end yields.",
    ),
    SignalRule(
        name="yield_selloff_momentum",
        metric="z_y10_change_5d",
        operator=">",
        threshold=1.0,
        interpretation="Yields are rising unusually quickly; trend may continue.",
    ),
    SignalRule(
        name="yield_rally_momentum",
        metric="z_y10_change_5d",
        operator="<",
        threshold=-1.0,
        interpretation="Yields are falling unusually quickly; trend may continue.",
    ),
    SignalRule(
        name="yield_selloff_exhaustion",
        metric="z_y10_change_20d",
        operator=">",
        threshold=2.0,
        interpretation="Yields have risen sharply and may be overextended.",
    ),
    SignalRule(
        name="yield_rally_exhaustion",
        metric="z_y10_change_20d",
        operator="<",
        threshold=-2.0,
        interpretation="Yields have fallen sharply and may be overextended.",
    ),
    SignalRule(
        name="curve_too_flat",
        metric="z_2s10s",
        operator="<",
        threshold=-2.0,
        interpretation="2s10s is unusually flat or inverted versus rolling history. The curve may be stretched and could steepen.",
    ),
    SignalRule(
        name="curve_too_steep",
        metric="z_2s10s",
        operator=">",
        threshold=2.0,
        interpretation="2s10s is unusually steep versus rolling history. The curve may be stretched and could flatten.",
    ),
    SignalRule(
        name="long_end_too_flat",
        metric="z_5s30s",
        operator="<",
        threshold=-2.0,
        interpretation="5s30s is unusually flat. Long-end yields are compressed relative to the belly; the long end may cheapen.",
    ),
    SignalRule(
        name="long_end_too_steep",
        metric="z_5s30s",
        operator=">",
        threshold=2.0,
        interpretation="5s30s is unusually steep. Long-end yields are high relative to the belly; the long end may richen.",
    ),
    SignalRule(
        name="five_y_cheap",
        metric="z_butterfly",
        operator=">",
        threshold=2.0,
        interpretation="5Y yield is high versus 2Y/10Y; belly is cheap.",
    ),
    SignalRule(
        name="five_y_rich",
        metric="z_butterfly",
        operator="<",
        threshold=-2.0,
        interpretation="5Y yield is low versus 2Y/10Y; belly is rich.",
    ),
)


MEAN_REVERSION_SIGNALS = {
    "yield_selloff_exhaustion",
    "yield_rally_exhaustion",
    "curve_too_flat",
    "curve_too_steep",
    "long_end_too_flat",
    "long_end_too_steep",
    "five_y_cheap",
    "five_y_rich",
}


SIGNAL_TO_TRADE: dict[str, SignalTradeMapping] = {
    "hot_CPI": build_multi_leg_mapping(
        trade_type="short_duration",
        label="Short Duration",
        legs=(("2Y", "short", 1.0), ("10Y", "short", 1.0)),
        rationale="Inflation is higher than expected. Fed may hike or stay hawkish, pushing yields higher and bond prices lower.",
    ),
    "cold_CPI": build_multi_leg_mapping(
        trade_type="long_duration",
        label="Long Duration",
        legs=(("2Y", "long", 1.0), ("10Y", "long", 1.0)),
        rationale="Inflation is lower than expected. Market may price more cuts or lower rates, pushing yields lower and bond prices higher.",
    ),
    "strong_payrolls": build_multi_leg_mapping(
        trade_type="short_duration",
        label="Short Duration",
        legs=(("2Y", "short", 1.0), ("10Y", "short", 1.0)),
        rationale="Strong employment/growth may support higher-for-longer rates, pushing yields higher and bond prices lower.",
    ),
    "weak_payrolls": build_multi_leg_mapping(
        trade_type="long_duration",
        label="Long Duration",
        legs=(("2Y", "long", 1.0), ("10Y", "long", 1.0)),
        rationale="Weak employment/growth may support lower rates or more cuts, pushing yields lower and bond prices higher.",
    ),
    "yield_selloff_momentum": build_outright_mapping(
        trade_type="short_duration_momentum",
        label="Short Duration Momentum",
        tenor="10Y",
        side="short",
        rationale="Yields are rising unusually quickly; test whether the selloff continues.",
    ),
    "yield_rally_momentum": build_outright_mapping(
        trade_type="long_duration_momentum",
        label="Long Duration Momentum",
        tenor="10Y",
        side="long",
        rationale="Yields are falling unusually quickly; test whether the rally continues.",
    ),
    "yield_selloff_exhaustion": build_outright_mapping(
        trade_type="long_duration_reversal",
        label="Long Duration Reversal",
        tenor="10Y",
        side="long",
        rationale="Yields have already risen sharply and may reverse lower.",
    ),
    "yield_rally_exhaustion": build_outright_mapping(
        trade_type="short_duration_reversal",
        label="Short Duration Reversal",
        tenor="10Y",
        side="short",
        rationale="Yields have already fallen sharply and may reverse higher.",
    ),
    "curve_too_flat": build_curve_mapping(
        trade_type="2s10s_steepener",
        label="2s10s Steepener",
        short_tenor="2Y",
        long_tenor="10Y",
        short_side="long",
        long_side="short",
        rationale="2s10s is unusually flat. Bet on steepening, meaning 10Y - 2Y increases.",
    ),
    "curve_too_steep": build_curve_mapping(
        trade_type="2s10s_flattener",
        label="2s10s Flattener",
        short_tenor="2Y",
        long_tenor="10Y",
        short_side="short",
        long_side="long",
        rationale="2s10s is unusually steep. Bet on flattening, meaning 10Y - 2Y decreases.",
    ),
    "long_end_too_flat": build_curve_mapping(
        trade_type="5s30s_steepener",
        label="5s30s Steepener",
        short_tenor="5Y",
        long_tenor="30Y",
        short_side="long",
        long_side="short",
        rationale="5s30s is unusually flat. Bet on steepening, meaning 30Y - 5Y increases.",
    ),
    "long_end_too_steep": build_curve_mapping(
        trade_type="5s30s_flattener",
        label="5s30s Flattener",
        short_tenor="5Y",
        long_tenor="30Y",
        short_side="short",
        long_side="long",
        rationale="5s30s is unusually steep. Bet on flattening, meaning 30Y - 5Y decreases.",
    ),
    "five_y_cheap": build_butterfly_mapping(
        trade_type="long_5y_butterfly",
        label="Long 5Y Butterfly",
        rationale="5Y yield is unusually high relative to 2Y and 10Y, so the 5Y bond is cheap. Bet on 5Y yield falling and 5Y bond price rising.",
        belly_side="long",
        wing_side="short",
    ),
    "five_y_rich": build_butterfly_mapping(
        trade_type="short_5y_butterfly",
        label="Short 5Y Butterfly",
        rationale="5Y yield is unusually low relative to 2Y and 10Y, so the 5Y bond is rich. Bet on 5Y yield rising and 5Y bond price falling.",
        belly_side="short",
        wing_side="long",
    ),
}


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    raise ValueError(f"Unsupported operator: {operator}")


def generate_signal_events(
    frame: pd.DataFrame,
    *,
    fresh_crossings_only: bool = True,
) -> list[SignalEvent]:
    events: list[SignalEvent] = []
    for rule in SIGNAL_RULES:
        if rule.metric not in frame.columns:
            continue
        series = frame[rule.metric]
        beyond_threshold = series.map(
            lambda value: False if pd.isna(value) else _compare(float(value), rule.operator, rule.threshold)
        )
        if rule.name in MEAN_REVERSION_SIGNALS:
            # An extreme is only a setup. Enter after the feature starts to
            # normalize by crossing back inside its threshold.
            active = beyond_threshold.shift(1, fill_value=False) & ~beyond_threshold
        elif fresh_crossings_only:
            active = beyond_threshold & ~beyond_threshold.shift(1, fill_value=False)
        else:
            active = beyond_threshold
        for date, value in series[active].items():
            events.append(
                SignalEvent(
                    date=pd.Timestamp(date),
                    signal_name=rule.name,
                    interpretation=rule.interpretation,
                    metric_name=rule.metric,
                    metric_value=float(value),
                    threshold=rule.threshold,
                )
            )
    return sorted(events, key=lambda event: (event.date, event.signal_name))
