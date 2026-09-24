from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .types import TradeLeg, TradeTemplate


@dataclass(frozen=True)
class TradeResult:
    signal_name: str
    template: TradeTemplate
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    gross_notional: float
    legs: tuple[TradeLeg, ...]
    metadata: dict[str, str]
