from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.regime import RegimeState, RegimeType
from src.evaluation.backtest import Side
from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Buy N std devs below rolling mean, sell at mean. Active in MEAN_REVERTING regime."""

    def __init__(
        self,
        window: int = 50,
        entry_std: float = 2.0,
        exit_std: float = 0.0,
        returns_lookback: int = 50,
    ):
        self.window = window
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.returns_lookback = returns_lookback

    def preferred_regimes(self) -> list[RegimeType]:
        return [RegimeType.MEAN_REVERTING]

    def generate_signal(
        self,
        data: pd.DataFrame,
        regime: RegimeState,
        usdt_balance: float,
        btc_position: float,
    ) -> Optional[tuple[Side, np.ndarray]]:
        if not self.is_active_regime(regime):
            return None

        if len(data) < self.window + 1:
            return None

        prices = data["close"].astype(float)
        current_price = prices.iloc[-1]

        rolling_mean = prices.rolling(self.window).mean().iloc[-1]
        rolling_std = prices.rolling(self.window).std().iloc[-1]

        if rolling_std == 0 or np.isnan(rolling_std):
            return None

        z_score = (current_price - rolling_mean) / rolling_std
        returns = prices.pct_change().dropna().tail(self.returns_lookback).values

        # Buy when price is entry_std below mean
        if z_score < -self.entry_std and btc_position == 0:
            return (Side.BUY, returns)

        # Sell when price reverts to mean (within exit_std)
        if abs(z_score) <= self.exit_std and btc_position > 0:
            return (Side.SELL, returns)

        # Also sell if price goes too far above (take profit)
        if z_score > self.entry_std and btc_position > 0:
            return (Side.SELL, returns)

        return None
