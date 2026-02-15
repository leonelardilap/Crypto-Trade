import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.regime import RegimeState, RegimeType


@pytest.fixture
def sample_ohlcv():
    """Synthetic 2000-row OHLCV data with realistic price movement."""
    np.random.seed(42)
    n = 2000
    # Random walk with drift
    log_returns = np.random.normal(0.0001, 0.01, n)
    prices = 50000 * np.exp(np.cumsum(log_returns))

    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices * (1 + np.random.normal(0, 0.002, n)),
        "volume": np.random.lognormal(10, 1, n),
        "close_time": dates + pd.Timedelta("59min"),
        "quote_asset_volume": np.random.lognormal(15, 1, n),
        "number_of_trades": np.random.randint(100, 10000, n),
        "taker_buy_base_volume": np.random.lognormal(9, 1, n),
        "taker_buy_quote_volume": np.random.lognormal(14, 1, n),
    })
    return df


@pytest.fixture
def sample_returns():
    """Fat-tailed returns drawn from t-distribution."""
    np.random.seed(123)
    return stats.t.rvs(df=5, loc=0.001, scale=0.02, size=500)


@pytest.fixture
def mock_regime():
    """A trending-up regime state with moderate confidence."""
    return RegimeState(
        regime=RegimeType.TRENDING_UP,
        confidence=0.7,
        timestamp=None,
        volatility=0.015,
        trend_strength=0.03,
        mean_reversion_score=-0.1,
    )
