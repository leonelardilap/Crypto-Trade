import numpy as np
import pandas as pd
import pytest

from src.analysis.regime import RegimeState, RegimeType
from src.evaluation.backtest import BacktestEngine, Side


def _buy_and_hold(data, regime, usdt, btc):
    """Simple buy-and-hold: buy once, never sell."""
    prices = data["close"].astype(float)
    returns = prices.pct_change().dropna().tail(50).values
    if btc == 0 and usdt > 0:
        return (Side.BUY, returns)
    return None


class TestBuyAndHold:
    def test_single_buy(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=10_000, fee_rate=0.001)
        result = engine.run(sample_ohlcv, _buy_and_hold, warmup_periods=100)

        # Should have at least 1 BUY trade
        buy_trades = [t for t in result.trades if t.side == Side.BUY]
        assert len(buy_trades) >= 1

    def test_equity_curve_produced(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(sample_ohlcv, _buy_and_hold, warmup_periods=100)
        assert not result.equity_curve.empty
        assert "equity" in result.equity_curve.columns

    def test_metrics_computed(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(sample_ohlcv, _buy_and_hold, warmup_periods=100)
        assert "sharpe" in result.metrics
        assert "total_return" in result.metrics
        assert "max_drawdown" in result.metrics


class TestFeeCorrectness:
    def test_fees_deducted(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=10_000, fee_rate=0.001)
        result = engine.run(sample_ohlcv, _buy_and_hold, warmup_periods=100)
        total_fees = result.metrics.get("total_fees", 0)
        if result.trades:
            assert total_fees > 0, "Fees should be positive when trades occur"
            # Fee per trade should be fee_rate * notional
            for t in result.trades:
                expected_fee = t.notional * 0.001
                assert abs(t.fee - expected_fee) < 0.01, \
                    f"Trade fee {t.fee} != expected {expected_fee}"


class TestBalanceConstraints:
    def test_balance_never_negative(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(sample_ohlcv, _buy_and_hold, warmup_periods=100)
        if not result.equity_curve.empty:
            assert (result.equity_curve["usdt"] >= -0.01).all(), "USDT balance went negative"
            assert (result.equity_curve["btc"] >= -0.0001).all(), "BTC position went negative"

    def test_no_trade_strategy(self, sample_ohlcv):
        """A strategy that never trades should preserve capital."""
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(sample_ohlcv, lambda *_: None, warmup_periods=100)
        assert len(result.trades) == 0
        assert result.metrics["total_fees"] == 0
