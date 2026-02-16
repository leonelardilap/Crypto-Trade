from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.regime import RegimeState, RegimeType
from src.evaluation.backtest import Side
from src.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Buy when price > fast SMA with volume confirmation; sell when price < slow SMA."""

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        volume_ma_period: int = 20,
        volume_threshold: float = 1.0,
        returns_lookback: int = 50,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.returns_lookback = returns_lookback

    def preferred_regimes(self) -> list[RegimeType]:
        return [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]

    def generate_signal(
        self,
        data: pd.DataFrame,
        regime: RegimeState,
        usdt_balance: float,
        btc_position: float,
    ) -> Optional[tuple[Side, np.ndarray]]:
        if not self.is_active_regime(regime):
            return None

        if len(data) < self.slow_period + 1:
            return None

        prices = data["close"].astype(float)
        current_price = prices.iloc[-1]

        fast_sma = prices.rolling(self.fast_period).mean().iloc[-1]
        slow_sma = prices.rolling(self.slow_period).mean().iloc[-1]

        # Volume confirmation
        volume_ok = True
        if "volume" in data.columns:
            volumes = data["volume"].astype(float)
            vol_ma = volumes.rolling(self.volume_ma_period).mean().iloc[-1]
            current_vol = volumes.iloc[-1]
            if vol_ma > 0:
                volume_ok = current_vol >= vol_ma * self.volume_threshold

        returns = prices.pct_change().dropna().tail(self.returns_lookback).values

        # Buy: price above fast SMA and volume confirms
        if current_price > fast_sma and volume_ok and btc_position == 0:
            return (Side.BUY, returns)

        # Sell: price below slow SMA
        if current_price < slow_sma and btc_position > 0:
            return (Side.SELL, returns)

        return None
