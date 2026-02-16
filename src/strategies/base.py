from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.regime import RegimeState, RegimeType
from src.evaluation.backtest import Side


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        regime: RegimeState,
        usdt_balance: float,
        btc_position: float,
    ) -> Optional[tuple[Side, np.ndarray]]:
        """Return (Side, returns_array) or None if no trade."""
        ...

    @abstractmethod
    def preferred_regimes(self) -> list[RegimeType]:
        """Regimes where this strategy should be active."""
        ...

    def is_active_regime(self, regime: RegimeState) -> bool:
        return regime.regime in self.preferred_regimes()
