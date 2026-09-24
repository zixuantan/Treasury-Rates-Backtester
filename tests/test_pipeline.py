import pandas as pd
from pathlib import Path

from rates_backtester import BacktestConfig, load_dataset, prepare_dataset, run_backtest, run_backtest_from_path


def test_real_treasury_yield_directory_loads():
    data_dir = Path(__file__).resolve().parents[1] / "Treasury Yield"
    loaded = load_dataset(data_dir)

    assert list(loaded.columns) == ["date", "y2", "y5", "y10", "y30"]
    assert loaded["date"].is_monotonic_increasing
    assert not loaded[["y2", "y5", "y10", "y30"]].isna().any().any()


def test_real_risk_sentiment_directory_loads():
    data_dir = Path(__file__).resolve().parents[1] / "Risk Sentiment Metrics"
    loaded = load_dataset(data_dir)

    assert list(loaded.columns) == ["date", "SPY", "VIX"]
    assert loaded["date"].is_monotonic_increasing
    assert not loaded[["SPY", "VIX"]].isna().any().any()


def test_market_context_directory_loads_supported_partial_set(tmp_path):
    context_dir = tmp_path / "Market Context"
    context_dir.mkdir()
    pd.DataFrame(
        {
            "observation_date": ["2024-01-02", "2024-01-03"],
            "T5YIE": [2.2, 2.25],
        }
    ).to_csv(context_dir / "T5YIE.csv", index=False)

    loaded = load_dataset(context_dir)
    assert list(loaded.columns) == ["date", "T5YIE"]
    assert loaded["T5YIE"].tolist() == [2.2, 2.25]


def test_workspace_root_loads_all_available_data():
    data_dir = Path(__file__).resolve().parents[1]
    loaded = load_dataset(data_dir)

    assert {"y2", "y5", "y10", "y30", "SPY", "VIX"}.issubset(set(loaded.columns))
    assert loaded["date"].is_monotonic_increasing


def test_workspace_root_backtest_runner():
    data_dir = Path(__file__).resolve().parents[1]
    result = run_backtest_from_path(data_dir, rolling_window=3, config=BacktestConfig(holding_period=1))

    assert result.portfolio.trade_count == len(result.trades)
    assert result.frame.index.is_monotonic_increasing


def test_dataset_loader_and_pipeline(tmp_path):
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="B"),
            "y2": [4.0, 4.0, 4.0, 4.0, 4.1, 4.2],
            "y5": [4.2, 4.2, 4.2, 4.2, 4.2, 4.2],
            "y10": [4.0, 4.0, 4.0, 4.0, 4.2, 4.4],
            "y30": [4.6, 4.6, 4.6, 4.6, 4.6, 4.6],
            "cpi_actual": [0.0, 1.0, 0.0, 5.0, 0.0, 0.0],
            "cpi_consensus": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "rates.csv"
    raw.to_csv(path, index=False)

    loaded = load_dataset(path)
    prepared = prepare_dataset(loaded, rolling_window=3)
    assert {"spread_2s10s", "z_cpi_surprise", "y10_change_1d"}.issubset(set(prepared.columns))

    result = run_backtest(loaded, rolling_window=3, config=BacktestConfig(holding_period=1))
    assert result.signal_events
    assert any(event.signal_name == "hot_CPI" for event in result.signal_events)
    assert result.trades
    assert result.portfolio.trade_count == len(result.trades)
    assert result.trades[0].metadata["trade_type"] == "short_duration"
