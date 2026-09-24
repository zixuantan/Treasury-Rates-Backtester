from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DURATION_ESTIMATES = {
    "2Y": 1.9,
    "5Y": 4.5,
    "10Y": 8.0,
    "30Y": 18.0,
}


@dataclass(frozen=True)
class TradeLegSpec:
    tenor: str
    side: str
    weight: float


@dataclass(frozen=True)
class SignalTradeMapping:
    trade_type: str
    label: str
    legs: tuple[TradeLegSpec, ...]
    rationale: str


def normalize_gross_weights(weights: Iterable[float], target_gross: float = 1.0) -> tuple[float, ...]:
    weights = tuple(float(weight) for weight in weights)
    gross = sum(abs(weight) for weight in weights)
    if gross == 0:
        raise ValueError("weights cannot all be zero")
    scale = target_gross / gross
    return tuple(weight * scale for weight in weights)


def _spec_leg(tenor: str, side: str, weight: float) -> TradeLegSpec:
    return TradeLegSpec(tenor=tenor, side=side, weight=weight)


def build_outright_mapping(trade_type: str, label: str, tenor: str, side: str, rationale: str) -> SignalTradeMapping:
    return SignalTradeMapping(trade_type=trade_type, label=label, legs=(_spec_leg(tenor, side, 1.0),), rationale=rationale)


def build_multi_leg_mapping(trade_type: str, label: str, legs: tuple[tuple[str, str, float], ...], rationale: str) -> SignalTradeMapping:
    return SignalTradeMapping(
        trade_type=trade_type,
        label=label,
        legs=tuple(_spec_leg(tenor, side, weight) for tenor, side, weight in legs),
        rationale=rationale,
    )


def build_curve_mapping(trade_type: str, label: str, short_tenor: str, long_tenor: str, short_side: str, long_side: str, rationale: str) -> SignalTradeMapping:
    short_weight = 1.0
    long_weight = DURATION_ESTIMATES[short_tenor] / DURATION_ESTIMATES[long_tenor]
    return SignalTradeMapping(
        trade_type=trade_type,
        label=label,
        legs=(
            _spec_leg(short_tenor, short_side, short_weight),
            _spec_leg(long_tenor, long_side, long_weight),
        ),
        rationale=rationale,
    )


def build_butterfly_mapping(trade_type: str, label: str, rationale: str, belly_side: str, wing_side: str) -> SignalTradeMapping:
    belly_weight = 1.0
    short_wing_weight = (DURATION_ESTIMATES["5Y"] / 2) / DURATION_ESTIMATES["2Y"]
    long_wing_weight = (DURATION_ESTIMATES["5Y"] / 2) / DURATION_ESTIMATES["10Y"]
    return SignalTradeMapping(
        trade_type=trade_type,
        label=label,
        legs=(
            _spec_leg("5Y", belly_side, belly_weight),
            _spec_leg("2Y", wing_side, short_wing_weight),
            _spec_leg("10Y", wing_side, long_wing_weight),
        ),
        rationale=rationale,
    )
