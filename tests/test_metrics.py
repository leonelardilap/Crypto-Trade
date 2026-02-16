import numpy as np
import pytest

from src.evaluation.metrics import MarketMetrics


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        sharpe = MarketMetrics.compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        assert sharpe > 0

    def test_zero_volatility(self):
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = MarketMetrics.compute_sharpe_ratio(returns, risk_free_rate=0.0)
        # All same → std=0 → return 0
        assert sharpe == 0.0

    def test_negative_returns(self):
        returns = np.array([-0.01, -0.02, -0.015, -0.01])
        sharpe = MarketMetrics.compute_sharpe_ratio(returns)
        assert sharpe < 0


class TestSortinoRatio:
    def test_all_positive(self):
        returns = np.array([0.01, 0.02, 0.03, 0.015])
        sortino = MarketMetrics.compute_sortino_ratio(returns)
        assert sortino == float("inf") or sortino > 0

    def test_mixed_returns(self):
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        sortino = MarketMetrics.compute_sortino_ratio(returns)
        assert sortino > 0


class TestCalmarRatio:
    def test_basic(self, sample_ohlcv):
        returns = sample_ohlcv["close"].pct_change().dropna().values
        calmar = MarketMetrics.compute_calmar_ratio(sample_ohlcv, returns, price_col="close")
        assert isinstance(calmar, float)


class TestWinRate:
    def test_all_winners(self):
        returns = np.array([0.01, 0.02, 0.03])
        assert MarketMetrics.compute_win_rate(returns) == 1.0

    def test_all_losers(self):
        returns = np.array([-0.01, -0.02, -0.03])
        assert MarketMetrics.compute_win_rate(returns) == 0.0

    def test_mixed(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        assert MarketMetrics.compute_win_rate(returns) == 0.5

    def test_empty(self):
        assert MarketMetrics.compute_win_rate(np.array([])) == 0.0


class TestProfitFactor:
    def test_profitable(self):
        returns = np.array([0.03, -0.01, 0.02, -0.005])
        pf = MarketMetrics.compute_profit_factor(returns)
        assert pf > 1.0, "Net profitable returns should have profit factor > 1"

    def test_no_losses(self):
        returns = np.array([0.01, 0.02])
        pf = MarketMetrics.compute_profit_factor(returns)
        assert pf == float("inf")
