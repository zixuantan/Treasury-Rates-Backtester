from datetime import date
import json

import pandas as pd

from rates_backtester import FREDClient, add_market_context_metrics
from rates_backtester.market_data import observations_to_series


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_observations_to_series_handles_fred_missing_values():
    series = observations_to_series(
        {
            "observations": [
                {"date": "2024-01-01", "value": "2.25"},
                {"date": "2024-01-02", "value": "."},
            ]
        },
        "T5YIE",
    )
    assert series.iloc[0] == 2.25
    assert pd.isna(series.iloc[1])


def test_fred_client_persists_and_reuses_cache(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return _FakeResponse({"observations": [{"date": "2024-01-01", "value": "2.25"}]})

    client = FREDClient("test-key", cache_dir=tmp_path, opener=opener)
    first = client.get_series(["T5YIE"], date(2024, 1, 1), date(2024, 1, 2))
    second = client.get_series(["T5YIE"], date(2024, 1, 1), date(2024, 1, 2))

    assert not first.from_cache
    assert second.from_cache
    assert len(calls) == 1
    assert second.data.loc[pd.Timestamp("2024-01-01"), "T5YIE"] == 2.25


def test_fred_client_can_request_a_single_as_of_vintage(tmp_path):
    urls = []

    def opener(request, timeout):
        urls.append(request.full_url)
        return _FakeResponse({"observations": [{"date": "2024-01-01", "value": "5.25"}]})

    client = FREDClient("test-key", cache_dir=tmp_path, opener=opener)
    result = client.get_series(
        ["EFFR"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        vintage_date=date(2024, 1, 3),
    )
    assert "realtime_start=2024-01-03" in urls[0]
    assert "realtime_end=2024-01-03" in urls[0]
    assert result.vintage_date == date(2024, 1, 3)


def test_market_context_metrics_are_optional_and_lag_safe():
    index = pd.date_range("2024-01-01", periods=12, freq="B")
    frame = pd.DataFrame(
        {
            "T5YIE": [2.0 + value**2 * 0.001 for value in range(12)],
            "DTWEXBGS": [100.0 + value for value in range(12)],
            "BAMLH0A0HYM2": [3.0 + value * 0.02 for value in range(12)],
            "BAMLC0A0CM": [1.0 + value * 0.01 for value in range(12)],
            "VIX": [15.0 + value * 0.2 for value in range(12)],
        },
        index=index,
    )
    enriched = add_market_context_metrics(frame, window=3)
    assert {
        "breakeven_5y_change_5d",
        "broad_dollar_return_5d",
        "hy_oas_change_5d_bp",
        "risk_regime_score",
    }.issubset(enriched.columns)
    assert pd.isna(enriched["z_breakeven_5y_change_5d"].iloc[7])
    assert pd.notna(enriched["z_breakeven_5y_change_5d"].iloc[8])
