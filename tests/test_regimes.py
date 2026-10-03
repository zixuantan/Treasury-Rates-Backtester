import pandas as pd

from rates_backtester import SignalEvent, apply_regime_filters, evaluate_regime_filter


def _event(name: str, metric: str, value: float) -> SignalEvent:
    return SignalEvent(
        date=pd.Timestamp("2024-01-02"),
        signal_name=name,
        interpretation="test",
        metric_name=metric,
        metric_value=value,
        threshold=1.0 if value > 0 else -1.0,
    )


def test_risk_on_context_vetoes_long_duration_trade():
    event = _event("yield_rally_momentum", "z_y10_change_5d", -2.0)
    frame = pd.DataFrame(
        {"yield_direction_score": [2], "z_ns_level": [0.0]},
        index=[event.date],
    )

    decision = evaluate_regime_filter(frame, event)

    assert not decision.passed
    assert decision.filter_name == "yield-direction score"


def test_neutral_context_does_not_block_trade():
    event = _event("yield_rally_momentum", "z_y10_change_5d", -2.0)
    frame = pd.DataFrame(
        {"yield_direction_score": [0], "z_ns_level": [0.0]},
        index=[event.date],
    )

    decision = evaluate_regime_filter(frame, event)

    assert decision.passed


def test_single_duration_vote_does_not_block_trade():
    event = _event("yield_rally_momentum", "z_y10_change_5d", -2.0)
    frame = pd.DataFrame(
        {"yield_direction_score": [1], "z_ns_level": [0.0]},
        index=[event.date],
    )

    decision = evaluate_regime_filter(frame, event)

    assert decision.passed


def test_nelson_siegel_shape_can_veto_butterfly_trade():
    event = _event("five_y_cheap", "z_butterfly", 2.5)
    frame = pd.DataFrame({"z_ns_curvature": [-1.5]}, index=[event.date])

    decision = evaluate_regime_filter(frame, event)

    assert not decision.passed
    assert decision.filter_name == "Nelson–Siegel shape"


def test_disabled_filter_accepts_every_candidate():
    event = _event("yield_rally_momentum", "z_y10_change_5d", -2.0)
    frame = pd.DataFrame({"yield_direction_score": [3]}, index=[event.date])

    accepted, decisions = apply_regime_filters(frame, [event], enabled=False)

    assert accepted == [event]
    assert decisions[0].passed
    assert decisions[0].filter_name == "disabled"
