from __future__ import annotations

from pathlib import Path
import pandas as pd

from .market_data import MARKET_CONTEXT_SERIES
from .types import YieldColumns


DEFAULT_DATE_COLUMN = "date"
YIELD_COLUMN_ALIASES = {
    "y2": "2Y",
    "y5": "5Y",
    "y10": "10Y",
    "y30": "30Y",
}
TREASURY_YIELD_FILES = {
    "DGS2.csv": "y2",
    "DGS5.csv": "y5",
    "DGS10.csv": "y10",
    "DGS30.csv": "y30",
}
OPTIONAL_TREASURY_YIELD_FILES = {
    "DGS1MO.csv": "y1m",
    "DGS3MO.csv": "y3m",
    "DGS6MO.csv": "y6m",
    "DGS1.csv": "y1",
}
RISK_SENTIMENT_FILES = {
    "SP500.csv": "SPY",
    "VIXCLS.csv": "VIX",
}
MARKET_CONTEXT_FILES = {f"{series_id}.csv": series_id for series_id in MARKET_CONTEXT_SERIES}


def _rename_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for source, target in YIELD_COLUMN_ALIASES.items():
        if source in renamed.columns and target not in renamed.columns:
            renamed[target] = renamed[source]
    return renamed


def load_treasury_yield_directory(directory: str | Path) -> pd.DataFrame:
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Treasury yield directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Treasury yield path must be a directory: {data_dir}")

    merged: pd.DataFrame | None = None
    for filename, column_name in TREASURY_YIELD_FILES.items():
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing treasury yield file: {file_path}")
        series = pd.read_csv(file_path, usecols=["observation_date", file_path.stem])
        series = series.rename(columns={"observation_date": DEFAULT_DATE_COLUMN, file_path.stem: column_name})
        series[DEFAULT_DATE_COLUMN] = pd.to_datetime(series[DEFAULT_DATE_COLUMN], utc=False)
        series[column_name] = pd.to_numeric(series[column_name], errors="coerce")
        if merged is None:
            merged = series
        else:
            merged = merged.merge(series, on=DEFAULT_DATE_COLUMN, how="inner")

    if merged is None or merged.empty:
        raise ValueError(f"No treasury yield data found in {data_dir}")

    for filename, column_name in OPTIONAL_TREASURY_YIELD_FILES.items():
        file_path = data_dir / filename
        if not file_path.exists():
            continue
        series = pd.read_csv(file_path, usecols=["observation_date", file_path.stem])
        series = series.rename(columns={"observation_date": DEFAULT_DATE_COLUMN, file_path.stem: column_name})
        series[DEFAULT_DATE_COLUMN] = pd.to_datetime(series[DEFAULT_DATE_COLUMN], utc=False)
        series[column_name] = pd.to_numeric(series[column_name], errors="coerce")
        merged = merged.merge(series, on=DEFAULT_DATE_COLUMN, how="left")

    merged = merged.dropna(subset=list(TREASURY_YIELD_FILES.values()))
    merged = merged.sort_values(DEFAULT_DATE_COLUMN).reset_index(drop=True)
    return merged


def load_market_context_directory(directory: str | Path) -> pd.DataFrame:
    """Load any supported daily FRED market-context CSVs in a directory."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Market context directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Market context path must be a directory: {data_dir}")

    merged: pd.DataFrame | None = None
    for filename, column_name in MARKET_CONTEXT_FILES.items():
        file_path = data_dir / filename
        if not file_path.exists():
            continue
        series = pd.read_csv(file_path, usecols=["observation_date", file_path.stem])
        series = series.rename(columns={"observation_date": DEFAULT_DATE_COLUMN, file_path.stem: column_name})
        series[DEFAULT_DATE_COLUMN] = pd.to_datetime(series[DEFAULT_DATE_COLUMN], utc=False)
        series[column_name] = pd.to_numeric(series[column_name], errors="coerce")
        merged = series if merged is None else merged.merge(series, on=DEFAULT_DATE_COLUMN, how="outer")

    if merged is None or merged.empty:
        raise ValueError(f"No supported market context data found in {data_dir}")
    return merged.sort_values(DEFAULT_DATE_COLUMN).reset_index(drop=True)


def load_risk_sentiment_directory(directory: str | Path) -> pd.DataFrame:
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Risk sentiment directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Risk sentiment path must be a directory: {data_dir}")

    merged: pd.DataFrame | None = None
    for filename, column_name in RISK_SENTIMENT_FILES.items():
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing risk sentiment file: {file_path}")
        series = pd.read_csv(file_path, usecols=["observation_date", file_path.stem])
        series = series.rename(columns={"observation_date": DEFAULT_DATE_COLUMN, file_path.stem: column_name})
        series[DEFAULT_DATE_COLUMN] = pd.to_datetime(series[DEFAULT_DATE_COLUMN], utc=False)
        series[column_name] = pd.to_numeric(series[column_name], errors="coerce")
        if merged is None:
            merged = series
        else:
            merged = merged.merge(series, on=DEFAULT_DATE_COLUMN, how="inner")

    if merged is None or merged.empty:
        raise ValueError(f"No risk sentiment data found in {data_dir}")

    merged = merged.dropna(subset=list(RISK_SENTIMENT_FILES.values()))
    merged = merged.sort_values(DEFAULT_DATE_COLUMN).reset_index(drop=True)
    return merged


def validate_yield_frame(frame: pd.DataFrame, columns: YieldColumns = YieldColumns()) -> pd.DataFrame:
    frame = _rename_aliases(frame)
    required = list(columns.required())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required yield columns: {missing}")
    if frame.empty:
        raise ValueError("Yield frame is empty")
    if DEFAULT_DATE_COLUMN in frame.columns:
        frame = frame.copy()
        frame[DEFAULT_DATE_COLUMN] = pd.to_datetime(frame[DEFAULT_DATE_COLUMN], utc=False)
        frame = frame.sort_values(DEFAULT_DATE_COLUMN).set_index(DEFAULT_DATE_COLUMN)
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Yield frame must have a DatetimeIndex or a 'date' column")
    else:
        frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Yield frame contains duplicate dates")
    return frame


def canonicalize_yield_frame(frame: pd.DataFrame, columns: YieldColumns = YieldColumns()) -> pd.DataFrame:
    validated = validate_yield_frame(frame, columns=columns)
    canonical = validated.copy()
    canonical.index = pd.DatetimeIndex(pd.to_datetime(canonical.index))
    canonical = canonical.sort_index()
    return canonical
