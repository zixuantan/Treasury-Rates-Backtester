from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

MARKET_CONTEXT_SERIES: dict[str, str] = {
    "T5YIE": "5-year breakeven inflation",
    "T10YIE": "10-year breakeven inflation",
    "T5YIFR": "5-year, 5-year-forward inflation expectation rate",
    "DTWEXBGS": "Nominal broad trade-weighted US dollar index",
    "BAMLH0A0HYM2": "US high-yield option-adjusted spread",
    "BAMLC0A0CM": "US corporate option-adjusted spread",
}


@dataclass(frozen=True)
class FREDResult:
    data: pd.DataFrame
    retrieved_at: pd.Timestamp
    from_cache: bool
    vintage_date: date | None = None


def observations_to_series(payload: dict[str, object], series_id: str) -> pd.Series:
    """Convert a FRED observations response to a numeric dated series."""
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("FRED response does not contain an observations list")
    records = [item for item in observations if isinstance(item, dict)]
    if not records:
        return pd.Series(dtype=float, name=series_id, index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame(records)
    if "date" not in frame.columns or "value" not in frame.columns:
        raise ValueError("FRED observations are missing date or value")
    index = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, name=series_id).loc[lambda value: value.index.notna()]
    series.index.name = "date"
    return series.sort_index()


class FREDClient:
    """Small Streamlit-independent FRED client with persistent CSV caching.

    Ordinary requests return the latest revised history. Passing vintage_date
    requests the single history known on that date. Complete release studies
    still need first-release timestamps and pre-release consensus data.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = ".cache/rates_backtester/fred",
        timeout: float = 30.0,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self.api_key = (api_key or os.getenv("FRED_API_KEY", "")).strip()
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._opener = opener

    def _cache_path(
        self,
        series_id: str,
        start_date: date,
        end_date: date,
        vintage_date: date | None,
    ) -> Path:
        safe_id = "".join(character for character in series_id if character.isalnum() or character in "_-")
        if safe_id != series_id:
            raise ValueError(f"Invalid FRED series ID: {series_id}")
        vintage_suffix = f"_vintage_{vintage_date.isoformat()}" if vintage_date else ""
        return self.cache_dir / f"{safe_id}_{start_date.isoformat()}_{end_date.isoformat()}{vintage_suffix}.csv"

    def _download_series(
        self,
        series_id: str,
        start_date: date,
        end_date: date,
        vintage_date: date | None,
    ) -> pd.Series:
        if not self.api_key:
            raise ValueError("FRED_API_KEY is required to download FRED data")
        parameters = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
        }
        if vintage_date is not None:
            parameters["realtime_start"] = vintage_date.isoformat()
            parameters["realtime_end"] = vintage_date.isoformat()
        query = urlencode(parameters)
        request = Request(f"{FRED_OBSERVATIONS_URL}?{query}", headers={"User-Agent": "rates-backtester/0.1"})
        with self._opener(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return observations_to_series(payload, series_id)

    def get_series(
        self,
        series_ids: Iterable[str],
        start_date: date,
        end_date: date,
        *,
        refresh: bool = False,
        vintage_date: date | None = None,
    ) -> FREDResult:
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        frames: list[pd.Series] = []
        all_cached = True
        for series_id in dict.fromkeys(series_ids):
            cache_path = self._cache_path(series_id, start_date, end_date, vintage_date)
            if cache_path.exists() and not refresh:
                cached = pd.read_csv(cache_path, parse_dates=["date"])
                series = pd.Series(
                    pd.to_numeric(cached[series_id], errors="coerce").to_numpy(),
                    index=pd.DatetimeIndex(cached["date"], name="date"),
                    name=series_id,
                )
            else:
                all_cached = False
                series = self._download_series(series_id, start_date, end_date, vintage_date)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                series.rename_axis("date").reset_index().to_csv(cache_path, index=False)
            frames.append(series)
        data = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
        data.index.name = "date"
        return FREDResult(
            data=data,
            retrieved_at=pd.Timestamp.now(tz="UTC"),
            from_cache=all_cached,
            vintage_date=vintage_date,
        )


def export_fred_csv_directory(frame: pd.DataFrame, directory: str | Path) -> list[Path]:
    """Write one FRED-compatible CSV per column for the directory loader."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    dated = frame.copy()
    if not isinstance(dated.index, pd.DatetimeIndex):
        raise ValueError("FRED export frame must have a DatetimeIndex")
    written: list[Path] = []
    for column in dated.columns:
        path = output_dir / f"{column}.csv"
        series = pd.DataFrame(
            {
                "observation_date": dated.index,
                column: pd.to_numeric(dated[column], errors="coerce"),
            }
        )
        series.to_csv(path, index=False)
        written.append(path)
    return written
