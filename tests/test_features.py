import pandas as pd
import pytest

from rates_backtester.features import (
    butterfly_2s5s10s,
    curve_spread,
    rolling_zscore,
)


def test_curve_spread():
    frame = pd.DataFrame({"2Y": [4.0, 4.1], "10Y": [4.5, 4.4]})
    spread = curve_spread(frame, "2Y", "10Y")
    assert spread.iloc[0] == pytest.approx(0.5)
    assert spread.iloc[1] == pytest.approx(0.3)


def test_butterfly():
    frame = pd.DataFrame({"2Y": [4.0], "5Y": [4.2], "10Y": [4.5]})
    butterfly = butterfly_2s5s10s(frame)
    assert butterfly.iloc[0] == pytest.approx(-0.05)


def test_rolling_zscore_is_shifted():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    zscore = rolling_zscore(series, 3)
    assert zscore.iloc[0] != zscore.iloc[0]
    assert zscore.iloc[1] != zscore.iloc[1]
    assert zscore.iloc[2] != zscore.iloc[2]
    assert zscore.iloc[3] == pytest.approx(2.0)

