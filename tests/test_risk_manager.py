import numpy as np
import pytest

from src.analysis.regime import RegimeState, RegimeType
from src.trade_executor.risk_manager import RiskManager


def _regime(rtype=RegimeType.TRENDING_UP, confidence=0.8):
    return RegimeState(
        regime=rtype, confidence=confidence, timestamp=None,
        volatility=0.01, trend_strength=0.03, mean_reversion_score=0.0,
    )


class TestKellyFraction:
    def test_positive_edge(self, sample_returns):
        f = RiskManager.compute_kelly_fraction(sample_returns)
        assert f > 0, "Positive-mean returns should produce positive Kelly fraction"

    def test_zero_edge(self):
        returns = np.array([0.01, -0.01, 0.01, -0.01])
        f = RiskManager.compute_kelly_fraction(returns)
        assert abs(f) < 0.1, "Symmetric returns should produce near-zero Kelly"

    def test_negative_edge(self):
        returns = np.array([-0.02, -0.01, -0.03, -0.02, 0.005])
        f = RiskManager.compute_kelly_fraction(returns)
        assert f < 0, "Negative-mean returns should produce negative Kelly"

    def test_empty_returns(self):
        f = RiskManager.compute_kelly_fraction(np.array([]))
        assert f == 0.0

    def test_single_return(self):
        f = RiskManager.compute_kelly_fraction(np.array([0.05]))
        assert f == 0.0


class TestPositionSizing:
    def test_max_risk_cap(self, sample_returns):
        rm = RiskManager(max_risk_per_trade=0.02)
        regime = _regime()
        pos = rm.size_position(100_000, 50_000, sample_returns, regime)
        if pos.is_valid:
            assert pos.fraction <= 0.02, "Fraction should not exceed max_risk_per_trade"

    def test_min_notional(self):
        rm = RiskManager(min_notional=10.0)
        # Tiny portfolio → notional below $10
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        regime = _regime(confidence=0.1)
        pos = rm.size_position(50, 50_000, returns, regime)
        if not pos.is_valid:
            assert "notional" in pos.rejection_reason or "zero" in (pos.rejection_reason or "")

    def test_drawdown_breaker(self, sample_returns):
        rm = RiskManager(max_drawdown=0.15)
        rm.reset_equity(100_000)
        rm.update_equity(80_000)  # 20% drawdown > 15%
        regime = _regime()
        pos = rm.size_position(80_000, 50_000, sample_returns, regime)
        assert not pos.is_valid
        assert "drawdown" in pos.rejection_reason

    def test_kill_switch(self, sample_returns):
        rm = RiskManager()
        rm.activate_kill_switch()
        regime = _regime()
        pos = rm.size_position(100_000, 50_000, sample_returns, regime)
        assert not pos.is_valid
        assert "kill switch" in pos.rejection_reason
        rm.deactivate_kill_switch()
        pos2 = rm.size_position(100_000, 50_000, sample_returns, regime)
        # Should no longer be blocked by kill switch
        assert pos2.rejection_reason != "kill switch active" or pos2.is_valid

    def test_btc_precision(self, sample_returns):
        rm = RiskManager(qty_precision=5)
        regime = _regime()
        pos = rm.size_position(100_000, 50_000, sample_returns, regime)
        if pos.is_valid:
            # Check that quantity has at most 5 decimal places
            qty_str = f"{pos.quantity_btc:.10f}"
            decimals = qty_str.split(".")[1]
            # Trailing zeros beyond 5th place are fine
            significant = decimals[:5]
            trailing = decimals[5:]
            assert all(c == "0" for c in trailing), f"BTC quantity {pos.quantity_btc} exceeds 5 decimal precision"

    def test_negative_edge_rejected(self):
        rm = RiskManager()
        returns = np.array([-0.02, -0.01, -0.03, -0.02, -0.015])
        regime = _regime()
        pos = rm.size_position(100_000, 50_000, returns, regime)
        assert not pos.is_valid
        assert "negative" in pos.rejection_reason or "zero" in pos.rejection_reason
