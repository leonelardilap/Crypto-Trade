import numpy as np
import pandas as pd
import pytest

from src.analysis.regime import RegimeDetector, RegimeType


def _make_df(prices, freq="1h"):
    n = len(prices)
    dates = pd.date_range("2023-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "open_time": dates,
        "close": prices,
    })


class TestRegimeDetection:
    def test_trending_up(self):
        """Steadily rising prices should detect TRENDING_UP."""
        np.random.seed(10)
        n = 200
        prices = 50000 + np.arange(n) * 100 + np.random.normal(0, 10, n)
        df = _make_df(prices)
        detector = RegimeDetector(trend_threshold=0.01)
        state = detector.detect(df)
        assert state.regime in (RegimeType.TRENDING_UP, RegimeType.HIGH_VOLATILITY), \
            f"Expected trending up, got {state.regime}"
        assert state.trend_strength > 0

    def test_trending_down(self):
        """Steadily falling prices should detect TRENDING_DOWN."""
        np.random.seed(11)
        n = 200
        prices = 70000 - np.arange(n) * 100 + np.random.normal(0, 10, n)
        df = _make_df(prices)
        detector = RegimeDetector(trend_threshold=0.01)
        state = detector.detect(df)
        assert state.regime in (RegimeType.TRENDING_DOWN, RegimeType.HIGH_VOLATILITY), \
            f"Expected trending down, got {state.regime}"
        assert state.trend_strength < 0

    def test_high_volatility(self):
        """Very noisy prices should detect HIGH_VOLATILITY."""
        np.random.seed(12)
        n = 200
        prices = 50000 + np.random.normal(0, 5000, n)
        prices = np.abs(prices)  # ensure positive
        df = _make_df(prices)
        detector = RegimeDetector(vol_high_threshold=0.01)
        state = detector.detect(df)
        assert state.regime == RegimeType.HIGH_VOLATILITY

    def test_confidence_range(self, sample_ohlcv):
        detector = RegimeDetector()
        state = detector.detect(sample_ohlcv)
        assert 0 <= state.confidence <= 1.0


class TestHurstExponent:
    def test_hurst_range(self):
        """Hurst exponent should be in (0, 1) for reasonable data."""
        np.random.seed(20)
        prices = pd.Series(50000 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500))))
        h = RegimeDetector.compute_hurst_exponent(prices)
        assert 0.0 < h < 1.0, f"Hurst {h} out of expected range"


class TestVarianceRatio:
    def test_random_walk_vr(self):
        """Variance ratio of random walk should be near 1.0."""
        np.random.seed(30)
        prices = pd.Series(50000 * np.exp(np.cumsum(np.random.normal(0, 0.01, 1000))))
        vr = RegimeDetector.compute_variance_ratio(prices, window=10)
        assert 0.5 < vr < 1.5, f"VR {vr} not near 1.0 for random walk"
