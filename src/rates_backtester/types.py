from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
class TradeTemplate(str, Enum):
    OUTRIGHT = "outright"
    CURVE_2S10S = "curve_2s10s"
    CURVE_5S30S = "curve_5s30s"
    BUTTERFLY_2S5S10S = "butterfly_2s5s10s"


@dataclass(frozen=True)
class YieldColumns:
    two_year: str = "2Y"
    five_year: str = "5Y"
    ten_year: str = "10Y"
    thirty_year: str = "30Y"

    def required(self) -> tuple[str, str, str, str]:
        return self.two_year, self.five_year, self.ten_year, self.thirty_year


@dataclass(frozen=True)
class BacktestConfig:
    holding_period: int = 5
    signal_lag: int = 1
    annualization_factor: int = 252
    gross_notional: float = 1_000_000.0
    use_regime_filters: bool = True


@dataclass(frozen=True)
class TradeLeg:
    tenor: str
    weight: float
    duration: float
