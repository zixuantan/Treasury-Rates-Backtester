from __future__ import annotations

import numpy as np
import pandas as pd

from .data import YIELD_COLUMN_ALIASES
from .types import YieldColumns


DEFAULT_ROLLING_WINDOW = 252
def _with_canonical_yields(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    for source, target in YIELD_COLUMN_ALIASES.items():
        if source in canonical.columns and target not in canonical.columns:
            canonical[target] = canonical[source]
    return canonical


def curve_spread(frame: pd.DataFrame, short_tenor: str, long_tenor: str, output_name: str | None = None) -> pd.Series:
    name = output_name or f"{short_tenor}_{long_tenor}_spread"
    return pd.Series(frame[long_tenor] - frame[short_tenor], index=frame.index, name=name)


def butterfly_2s5s10s(frame: pd.DataFrame, columns: YieldColumns = YieldColumns(), output_name: str = "2s5s10s_butterfly") -> pd.Series:
    two = frame[columns.two_year]
    five = frame[columns.five_year]
    ten = frame[columns.ten_year]
    return pd.Series(five - (two + ten) / 2.0, index=frame.index, name=output_name)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least 2")
    shifted = series.shift(1)
    mean = shifted.rolling(window, min_periods=window).mean()
    std = shifted.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def add_required_metrics(frame: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW, columns: YieldColumns = YieldColumns()) -> pd.DataFrame:
    enriched = _with_canonical_yields(frame)

    spread_2s10s = curve_spread(enriched, columns.two_year, columns.ten_year, "spread_2s10s")
    spread_5s30s = curve_spread(enriched, columns.five_year, columns.thirty_year, "spread_5s30s")
    fly_2s5s10s = butterfly_2s5s10s(enriched, columns, "fly_2s5s10s")

    enriched["spread_2s10s"] = spread_2s10s
    enriched["spread_5s30s"] = spread_5s30s
    enriched["fly_2s5s10s"] = fly_2s5s10s
    enriched["z_2s10s"] = rolling_zscore(spread_2s10s, window)
    enriched["z_5s30s"] = rolling_zscore(spread_5s30s, window)
    enriched["z_butterfly"] = rolling_zscore(fly_2s5s10s, window)

    enriched["y10_change_1d"] = enriched[columns.ten_year].diff(1)
    enriched["y10_change_5d"] = enriched[columns.ten_year].diff(5)
    enriched["y10_change_20d"] = enriched[columns.ten_year].diff(20)
    enriched["z_y10_change_5d"] = rolling_zscore(enriched["y10_change_5d"], window)
    enriched["z_y10_change_20d"] = rolling_zscore(enriched["y10_change_20d"], window)

    if "cpi_actual" in enriched.columns and "cpi_consensus" in enriched.columns:
        enriched["cpi_surprise"] = enriched["cpi_actual"] - enriched["cpi_consensus"]
        enriched["z_cpi_surprise"] = rolling_zscore(enriched["cpi_surprise"], window)
    if "nfp_actual" in enriched.columns and "nfp_consensus" in enriched.columns:
        enriched["nfp_surprise"] = enriched["nfp_actual"] - enriched["nfp_consensus"]
        enriched["z_nfp_surprise"] = rolling_zscore(enriched["nfp_surprise"], window)

    return enriched


def add_market_context_metrics(frame: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW) -> pd.DataFrame:
    """Add optional, lag-safe market-context metrics when source columns exist.

    Daily market observations are measured through the current close. The
    backtest's default one-day signal lag means they are traded no earlier
    than the following row. Rolling z-scores exclude the current observation.
    """
    enriched = frame.copy()
    change_metrics: dict[str, pd.Series] = {}

    for source, prefix in {
        "T5YIE": "breakeven_5y",
        "T10YIE": "breakeven_10y",
        "T5YIFR": "breakeven_5y5y",
    }.items():
        if source in enriched.columns:
            change_metrics[f"{prefix}_change_5d"] = enriched[source].diff(5)

    if "DTWEXBGS" in enriched.columns:
        change_metrics["broad_dollar_return_5d"] = enriched["DTWEXBGS"].pct_change(5, fill_method=None) * 100.0
    if "BAMLH0A0HYM2" in enriched.columns:
        change_metrics["hy_oas_change_5d_bp"] = enriched["BAMLH0A0HYM2"].diff(5) * 100.0
    if "BAMLC0A0CM" in enriched.columns:
        change_metrics["ig_oas_change_5d_bp"] = enriched["BAMLC0A0CM"].diff(5) * 100.0

    vix_column = "VIX" if "VIX" in enriched.columns else "VIXCLS" if "VIXCLS" in enriched.columns else None
    if vix_column is not None:
        change_metrics["vix_change_5d"] = enriched[vix_column].diff(5)
    if "SPY" in enriched.columns:
        change_metrics["spy_return_5d"] = enriched["SPY"].pct_change(5, fill_method=None) * 100.0

    for name, series in change_metrics.items():
        enriched[name] = series
        enriched[f"z_{name}"] = rolling_zscore(series, window)

    vote_inputs = {
        "hy": enriched.get("hy_oas_change_5d_bp"),
        "ig": enriched.get("ig_oas_change_5d_bp"),
        "vix": enriched.get("vix_change_5d"),
        "dollar": enriched.get("broad_dollar_return_5d"),
        "spy": enriched.get("spy_return_5d"),
    }
    if any(series is not None for series in vote_inputs.values()):
        risk_on = pd.Series(0, index=enriched.index, dtype=int)
        risk_off = pd.Series(0, index=enriched.index, dtype=int)
        available_votes = pd.Series(0, index=enriched.index, dtype=int)
        thresholds = {
            "hy": (-5.0, 5.0),
            "ig": (-5.0, 5.0),
            "vix": (-1.0, 1.0),
            "dollar": (-0.5, 0.5),
            # Positive equity returns are risk-on, unlike the other inputs
            # where a negative change is risk-on.
            "spy": (1.0, -1.0),
        }
        for name, series in vote_inputs.items():
            if series is None:
                continue
            available_votes = available_votes + series.notna().astype(int)
            on_threshold, off_threshold = thresholds[name]
            if name == "spy":
                risk_on = risk_on + series.ge(on_threshold).fillna(False).astype(int)
                risk_off = risk_off + series.le(off_threshold).fillna(False).astype(int)
            else:
                risk_on = risk_on + series.le(on_threshold).fillna(False).astype(int)
                risk_off = risk_off + series.ge(off_threshold).fillna(False).astype(int)
        has_context = available_votes.gt(0)
        enriched["risk_on_votes"] = risk_on.where(has_context).astype("Int64")
        enriched["risk_off_votes"] = risk_off.where(has_context).astype("Int64")
        enriched["risk_regime_score"] = (risk_on - risk_off).where(has_context).astype("Int64")

    breakeven_changes = [
        enriched[column]
        for column in (
            "breakeven_5y_change_5d",
            "breakeven_10y_change_5d",
            "breakeven_5y5y_change_5d",
        )
        if column in enriched.columns
    ]
    if breakeven_changes:
        inflation_up = pd.Series(0, index=enriched.index, dtype=int)
        inflation_down = pd.Series(0, index=enriched.index, dtype=int)
        available = pd.Series(0, index=enriched.index, dtype=int)
        for series in breakeven_changes:
            available = available + series.notna().astype(int)
            inflation_up = inflation_up + series.ge(0.05).fillna(False).astype(int)
            inflation_down = inflation_down + series.le(-0.05).fillna(False).astype(int)
        has_inflation = available.gt(0)
        enriched["inflation_up_votes"] = inflation_up.where(has_inflation).astype("Int64")
        enriched["inflation_down_votes"] = inflation_down.where(has_inflation).astype("Int64")
        enriched["inflation_regime_score"] = (inflation_up - inflation_down).where(has_inflation).astype("Int64")

    score_columns = [
        enriched[column]
        for column in ("risk_regime_score", "inflation_regime_score")
        if column in enriched.columns
    ]
    if score_columns:
        score_frame = pd.concat(score_columns, axis=1)
        enriched["duration_regime_score"] = score_frame.sum(axis=1, min_count=1).astype("Int64")

    return enriched
