import logging
import math
from dataclasses import dataclass

import numpy as np

from src.analysis.regime import RegimeType, RegimeState
from src.config import MAX_RISK_PER_TRADE, KELLY_FRACTION, MAX_POSITION_PCT
from src.utils.constants import BTCUSDT_INFO

logger = logging.getLogger(__name__)

# Regime scaling factors
REGIME_MULTIPLIERS = {
    RegimeType.TRENDING_UP: 1.0,
    RegimeType.TRENDING_DOWN: 1.0,
    RegimeType.MEAN_REVERTING: 0.5,
    RegimeType.HIGH_VOLATILITY: 0.25,
    RegimeType.UNKNOWN: 0.1,
}


@dataclass
class PositionSize:
    fraction: float
    raw_kelly: float
    notional_usd: float
    quantity_btc: float
    risk_pct: float
    regime: RegimeType
    is_valid: bool
    rejection_reason: str | None = None


class RiskManager:
    """Kelly criterion position sizing with drawdown circuit breaker and kill switch."""

    def __init__(
        self,
        kelly_fraction: float = KELLY_FRACTION,
        max_risk_per_trade: float = MAX_RISK_PER_TRADE,
        max_position_pct: float = MAX_POSITION_PCT,
        max_drawdown: float = 0.15,
        min_notional: float = BTCUSDT_INFO.min_notional,
        qty_precision: int = BTCUSDT_INFO.qty_precision,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown
        self.min_notional = min_notional
        self.qty_precision = qty_precision
        self._kill_switch = False
        self._peak_equity = 0.0
        self._current_equity = 0.0

    # ---- Kelly ----

    @staticmethod
    def compute_kelly_fraction(returns: np.ndarray) -> float:
        """f* = E[phi] / E[phi^2] from the README."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        e_phi = np.mean(returns)
        e_phi2 = np.mean(returns ** 2)
        if e_phi2 == 0:
            return 0.0
        f_star = e_phi / e_phi2
        return float(f_star)

    def adjust_kelly_by_regime(self, raw_kelly: float, regime: RegimeState) -> float:
        """Scale Kelly fraction by regime type and confidence."""
        multiplier = REGIME_MULTIPLIERS.get(regime.regime, 0.1)
        # Blend multiplier with confidence
        effective = multiplier * regime.confidence
        adjusted = raw_kelly * effective * self.kelly_fraction
        return max(0.0, adjusted)

    # ---- Position sizing ----

    def size_position(
        self,
        portfolio_value: float,
        current_price: float,
        returns: np.ndarray,
        regime: RegimeState,
        existing_position_usd: float = 0.0,
    ) -> PositionSize:
        """Compute position size with all safety checks."""
        raw_kelly = self.compute_kelly_fraction(returns)

        # Kill switch
        if self._kill_switch:
            return PositionSize(
                fraction=0.0, raw_kelly=raw_kelly, notional_usd=0.0,
                quantity_btc=0.0, risk_pct=0.0, regime=regime.regime,
                is_valid=False, rejection_reason="kill switch active",
            )

        # Drawdown circuit breaker
        self.update_equity(portfolio_value)
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown > self.max_drawdown:
                return PositionSize(
                    fraction=0.0, raw_kelly=raw_kelly, notional_usd=0.0,
                    quantity_btc=0.0, risk_pct=0.0, regime=regime.regime,
                    is_valid=False,
                    rejection_reason=f"drawdown {drawdown:.1%} exceeds {self.max_drawdown:.1%} limit",
                )

        # Negative edge
        if raw_kelly <= 0:
            return PositionSize(
                fraction=0.0, raw_kelly=raw_kelly, notional_usd=0.0,
                quantity_btc=0.0, risk_pct=0.0, regime=regime.regime,
                is_valid=False, rejection_reason="negative or zero edge",
            )

        fraction = self.adjust_kelly_by_regime(raw_kelly, regime)

        # Cap at max risk
        fraction = min(fraction, self.max_risk_per_trade)

        # Cap total position including existing
        max_new = max(0.0, self.max_position_pct - existing_position_usd / portfolio_value) if portfolio_value > 0 else 0.0
        fraction = min(fraction, max_new)

        notional_usd = fraction * portfolio_value

        # Min notional check
        if notional_usd < self.min_notional:
            return PositionSize(
                fraction=fraction, raw_kelly=raw_kelly, notional_usd=notional_usd,
                quantity_btc=0.0, risk_pct=fraction, regime=regime.regime,
                is_valid=False,
                rejection_reason=f"notional ${notional_usd:.2f} below min ${self.min_notional}",
            )

        # BTC quantity with precision rounding
        quantity_btc = math.floor(notional_usd / current_price * 10**self.qty_precision) / 10**self.qty_precision

        if quantity_btc <= 0:
            return PositionSize(
                fraction=fraction, raw_kelly=raw_kelly, notional_usd=notional_usd,
                quantity_btc=0.0, risk_pct=fraction, regime=regime.regime,
                is_valid=False, rejection_reason="quantity rounds to zero",
            )

        return PositionSize(
            fraction=fraction,
            raw_kelly=raw_kelly,
            notional_usd=quantity_btc * current_price,
            quantity_btc=quantity_btc,
            risk_pct=fraction,
            regime=regime.regime,
            is_valid=True,
        )

    # ---- Equity tracking ----

    def update_equity(self, equity: float):
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def reset_equity(self, equity: float):
        self._peak_equity = equity
        self._current_equity = equity

    # ---- Kill switch ----

    def activate_kill_switch(self):
        self._kill_switch = True
        logger.warning("Kill switch ACTIVATED — all sizing blocked")

    def deactivate_kill_switch(self):
        self._kill_switch = False
        logger.info("Kill switch deactivated")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch
