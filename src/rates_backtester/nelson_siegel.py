from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DECAY = 0.6

# Each tenor can be supplied using the FRED identifier, the package's
# canonical label, or the short alias produced by the directory loader.
TENOR_COLUMN_ALIASES: dict[float, tuple[str, ...]] = {
    1 / 12: ("DGS1MO", "1M", "y1m"),
    3 / 12: ("DGS3MO", "3M", "y3m"),
    6 / 12: ("DGS6MO", "6M", "y6m"),
    1.0: ("DGS1", "1Y", "y1"),
    2.0: ("DGS2", "2Y", "y2"),
    5.0: ("DGS5", "5Y", "y5"),
    10.0: ("DGS10", "10Y", "y10"),
    30.0: ("DGS30", "30Y", "y30"),
}


@dataclass(frozen=True)
class NelsonSiegelFit:
    level: float
    slope: float
    curvature: float
    rmse_bp: float
    tenor_count: int


def nelson_siegel_curve(
    tenors: Iterable[float] | np.ndarray,
    level: float,
    slope: float,
    curvature: float,
    *,
    decay: float = DEFAULT_DECAY,
) -> np.ndarray:
    """Evaluate a fixed-decay Nelson-Siegel curve with tenors in years."""
    if decay <= 0:
        raise ValueError("decay must be positive")
    tau = np.asarray(tuple(tenors) if not isinstance(tenors, np.ndarray) else tenors, dtype=float)
    x = tau / decay
    with np.errstate(divide="ignore", invalid="ignore"):
        loading_slope = (1.0 - np.exp(-x)) / x
    loading_slope = np.where(np.isfinite(loading_slope), loading_slope, 1.0)
    loading_curvature = loading_slope - np.exp(-x)
    return level + slope * loading_slope + curvature * loading_curvature


def fit_nelson_siegel_curve(
    tenors: Iterable[float],
    yields: Iterable[float],
    *,
    decay: float = DEFAULT_DECAY,
) -> NelsonSiegelFit:
    """Fit fixed-decay factors using linear least squares."""
    tau = np.asarray(tuple(tenors), dtype=float)
    observed = np.asarray(tuple(yields), dtype=float)
    valid = np.isfinite(tau) & np.isfinite(observed)
    tau = tau[valid]
    observed = observed[valid]
    if observed.size < 4:
        raise ValueError("at least four valid tenors are required")

    x = tau / decay
    loading_slope = (1.0 - np.exp(-x)) / x
    loading_curvature = loading_slope - np.exp(-x)
    design = np.column_stack((np.ones_like(tau), loading_slope, loading_curvature))
    parameters, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank < 3:
        raise ValueError("yield curve does not identify all Nelson-Siegel factors")
    fitted = design @ parameters
    rmse_bp = float(np.sqrt(np.mean((observed - fitted) ** 2)) * 100.0)
    return NelsonSiegelFit(
        level=float(parameters[0]),
        slope=float(parameters[1]),
        curvature=float(parameters[2]),
        rmse_bp=rmse_bp,
        tenor_count=int(observed.size),
    )


def _available_tenors(frame: pd.DataFrame) -> list[tuple[float, str]]:
    available: list[tuple[float, str]] = []
    for tenor, aliases in TENOR_COLUMN_ALIASES.items():
        column = next((alias for alias in aliases if alias in frame.columns), None)
        if column is not None:
            available.append((tenor, column))
    return available


def add_nelson_siegel_factors(frame: pd.DataFrame, *, decay: float = DEFAULT_DECAY) -> pd.DataFrame:
    """Append daily NS factors; rows with fewer than four yields remain NaN."""
    enriched = frame.copy()
    available = _available_tenors(enriched)
    output_columns = ("ns_level", "ns_slope", "ns_curvature", "ns_rmse_bp", "ns_tenor_count")
    for column in output_columns:
        enriched[column] = np.nan
    if len(available) < 4:
        return enriched

    tenors = np.asarray([tenor for tenor, _ in available], dtype=float)
    columns = [column for _, column in available]
    for date, row in enriched.loc[:, columns].iterrows():
        values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 4:
            continue
        try:
            fit = fit_nelson_siegel_curve(tenors[valid], values[valid], decay=decay)
        except ValueError:
            continue
        enriched.loc[date, list(output_columns)] = (
            fit.level,
            fit.slope,
            fit.curvature,
            fit.rmse_bp,
            fit.tenor_count,
        )
    return enriched
