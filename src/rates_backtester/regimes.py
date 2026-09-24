from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .signals import SIGNAL_TO_TRADE, SignalEvent
from .trades import DURATION_ESTIMATES, normalize_gross_weights


@dataclass(frozen=True)
class RegimeDecision:
    event: SignalEvent
    passed: bool
    filter_name: str
    score: float | None
    reason: str


_SHAPE_CONFIRMATION = {
    "curve_too_flat": "z_ns_slope",
    "curve_too_steep": "z_ns_slope",
    "long_end_too_flat": "z_ns_slope",
    "long_end_too_steep": "z_ns_slope",
    "five_y_cheap": "z_ns_curvature",
    "five_y_rich": "z_ns_curvature",
}

_LEVEL_CONFIRMATION = {
    "yield_selloff_momentum",
    "yield_rally_momentum",
    "yield_selloff_exhaustion",
    "yield_rally_exhaustion",
}


def _finite_value(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _duration_direction(signal_name: str) -> float:
    mapping = SIGNAL_TO_TRADE[signal_name]
    weights = normalize_gross_weights(leg.weight for leg in mapping.legs)
    return sum(
        (weight if leg.side == "long" else -weight) * DURATION_ESTIMATES[leg.tenor]
        for leg, weight in zip(mapping.legs, weights)
    )


def evaluate_regime_filter(frame: pd.DataFrame, event: SignalEvent) -> RegimeDecision:
    """Veto a signal only when available context clearly contradicts it.

    Market context is observed on the signal row and remains tradable only
    after the configured execution lag. Missing or neutral context is
    deliberately fail-open so optional data cannot silently disable a model.
    """
    if event.date not in frame.index:
        return RegimeDecision(event, True, "none", None, "Signal date is absent from the prepared frame; no veto applied.")
    row = frame.loc[event.date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    duration_direction = _duration_direction(event.signal_name)
    if abs(duration_direction) > 1e-9:
        context_score = _finite_value(row, "duration_regime_score")
        if context_score is not None and context_score != 0:
            # Positive context favors rising yields/short duration; negative
            # context favors falling yields/long duration.
            if duration_direction * context_score > 0:
                return RegimeDecision(
                    event,
                    False,
                    "duration regime",
                    context_score,
                    "Risk and inflation context conflicts with the trade's duration direction.",
                )

        if event.signal_name in _LEVEL_CONFIRMATION:
            level_score = _finite_value(row, "z_ns_level")
            if level_score is not None and abs(level_score) >= 1.0 and level_score * event.metric_value < 0:
                return RegimeDecision(
                    event,
                    False,
                    "Nelson–Siegel level",
                    level_score,
                    "The fitted curve level strongly contradicts the yield-move signal.",
                )

        if context_score is not None:
            return RegimeDecision(
                event,
                True,
                "duration regime",
                context_score,
                "Available context is neutral or does not conflict with the trade's duration direction.",
            )

    confirmation_field = _SHAPE_CONFIRMATION.get(event.signal_name)
    if confirmation_field is not None:
        shape_score = _finite_value(row, confirmation_field)
        if shape_score is not None and abs(shape_score) >= 1.0 and shape_score * event.metric_value < 0:
            return RegimeDecision(
                event,
                False,
                "Nelson–Siegel shape",
                shape_score,
                f"{confirmation_field} strongly contradicts the curve or butterfly signal.",
            )
        if shape_score is not None:
            return RegimeDecision(
                event,
                True,
                "Nelson–Siegel shape",
                shape_score,
                f"{confirmation_field} is neutral or agrees with the curve or butterfly signal.",
            )

    return RegimeDecision(event, True, "none", None, "No relevant context was available; no veto applied.")


def apply_regime_filters(
    frame: pd.DataFrame,
    events: list[SignalEvent],
    *,
    enabled: bool = True,
) -> tuple[list[SignalEvent], tuple[RegimeDecision, ...]]:
    if not enabled:
        decisions = tuple(
            RegimeDecision(event, True, "disabled", None, "Regime filtering is disabled.")
            for event in events
        )
        return list(events), decisions
    decisions = tuple(evaluate_regime_filter(frame, event) for event in events)
    accepted = [decision.event for decision in decisions if decision.passed]
    return accepted, decisions
