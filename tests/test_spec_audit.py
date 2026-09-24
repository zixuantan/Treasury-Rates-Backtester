import pandas as pd
import pytest

from rates_backtester import (
    SIGNAL_RULES,
    SIGNAL_TO_TRADE,
    add_required_metrics,
    generate_signal_events,
)
from rates_backtester.features import rolling_zscore


EXPECTED_METRICS = {
    "spread_2s10s",
    "spread_5s30s",
    "fly_2s5s10s",
    "z_2s10s",
    "z_5s30s",
    "z_butterfly",
    "y10_change_1d",
    "y10_change_5d",
    "y10_change_20d",
    "z_y10_change_5d",
    "z_y10_change_20d",
    "cpi_surprise",
    "nfp_surprise",
    "z_cpi_surprise",
    "z_nfp_surprise",
}

EXPECTED_SIGNALS = {
    "hot_CPI",
    "cold_CPI",
    "strong_payrolls",
    "weak_payrolls",
    "yield_selloff_momentum",
    "yield_rally_momentum",
    "yield_selloff_exhaustion",
    "yield_rally_exhaustion",
    "curve_too_flat",
    "curve_too_steep",
    "long_end_too_flat",
    "long_end_too_steep",
    "five_y_cheap",
    "five_y_rich",
}

EXPECTED_MAPPINGS = EXPECTED_SIGNALS


def _metric_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="B"),
            "y2": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "y5": [4.2, 4.2, 4.2, 4.2, 4.2, 4.2],
            "y10": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            "y30": [4.6, 4.6, 4.6, 4.6, 4.6, 4.6],
            "cpi_actual": [3.2, 3.1, 3.4, 3.0, 3.3, 3.2],
            "cpi_consensus": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "nfp_actual": [200, 210, 220, 230, 240, 250],
            "nfp_consensus": [190, 200, 210, 220, 230, 240],
        }
    )
    return add_required_metrics(frame, window=3)


def test_all_required_metrics_exist():
    frame = _metric_frame()
    assert EXPECTED_METRICS.issubset(set(frame.columns))


def test_all_required_signals_exist():
    frame = pd.DataFrame(
        {
            "z_cpi_surprise": [-1.5, 1.5],
            "z_nfp_surprise": [-1.5, 1.5],
            "z_y10_change_5d": [-1.25, 1.25],
            "z_y10_change_20d": [-2.5, 2.5],
            "z_2s10s": [-2.5, 2.5],
            "z_5s30s": [-2.5, 2.5],
            "z_butterfly": [-2.5, 2.5],
        },
        index=pd.to_datetime(["2024-01-04", "2024-01-05"]),
    )
    events = generate_signal_events(frame)
    assert {event.signal_name for event in events} == EXPECTED_SIGNALS
    assert all(event.interpretation for event in events)


def test_all_required_mappings_exist():
    assert set(SIGNAL_TO_TRADE) == EXPECTED_MAPPINGS


def test_curve_trade_leg_directions():
    hot_cpi = SIGNAL_TO_TRADE["hot_CPI"]
    cold_cpi = SIGNAL_TO_TRADE["cold_CPI"]
    strong_payrolls = SIGNAL_TO_TRADE["strong_payrolls"]
    weak_payrolls = SIGNAL_TO_TRADE["weak_payrolls"]
    steepener = SIGNAL_TO_TRADE["curve_too_flat"]
    flattener = SIGNAL_TO_TRADE["curve_too_steep"]
    long_end_steepener = SIGNAL_TO_TRADE["long_end_too_flat"]
    long_end_flattener = SIGNAL_TO_TRADE["long_end_too_steep"]

    assert [(leg.tenor, leg.side) for leg in hot_cpi.legs] == [("2Y", "short"), ("10Y", "short")]
    assert [(leg.tenor, leg.side) for leg in cold_cpi.legs] == [("2Y", "long"), ("10Y", "long")]
    assert [(leg.tenor, leg.side) for leg in strong_payrolls.legs] == [("2Y", "short"), ("10Y", "short")]
    assert [(leg.tenor, leg.side) for leg in weak_payrolls.legs] == [("2Y", "long"), ("10Y", "long")]
    assert [(leg.tenor, leg.side) for leg in steepener.legs] == [("2Y", "long"), ("10Y", "short")]
    assert [(leg.tenor, leg.side) for leg in flattener.legs] == [("2Y", "short"), ("10Y", "long")]
    assert [(leg.tenor, leg.side) for leg in long_end_steepener.legs] == [("5Y", "short"), ("30Y", "long")]
    assert [(leg.tenor, leg.side) for leg in long_end_flattener.legs] == [("5Y", "long"), ("30Y", "short")]


def test_butterfly_trade_leg_directions():
    cheap = SIGNAL_TO_TRADE["five_y_cheap"]
    rich = SIGNAL_TO_TRADE["five_y_rich"]

    assert [(leg.tenor, leg.side) for leg in cheap.legs] == [("5Y", "long"), ("2Y", "short"), ("10Y", "short")]
    assert [(leg.tenor, leg.side) for leg in rich.legs] == [("5Y", "short"), ("2Y", "long"), ("10Y", "long")]
    assert cheap.legs[0].weight == pytest.approx(1.0)
    assert cheap.legs[1].weight == pytest.approx((4.5 / 2) / 1.9)
    assert cheap.legs[2].weight == pytest.approx((4.5 / 2) / 8.0)


def test_no_lookahead_rolling_zscores():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "y2": [0.0, 0.0, 0.0, 0.0],
            "y5": [0.0, 0.0, 0.0, 0.0],
            "y10": [1.0, 2.0, 3.0, 4.0],
            "y30": [0.0, 0.0, 0.0, 0.0],
        }
    )
    metrics = add_required_metrics(frame, window=3)
    assert metrics.loc[metrics.index[2], "z_2s10s"] != metrics.loc[metrics.index[2], "z_2s10s"]
    assert metrics.loc[metrics.index[3], "z_2s10s"] == pytest.approx(2.0)
