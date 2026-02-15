import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fetch.data_cache import DataCache, OHLCV_COLUMNS


def _make_kline_df(n=100, start_price=50000.0, freq="1h"):
    """Create a minimal kline DataFrame for testing."""
    np.random.seed(99)
    dates = pd.date_range("2023-06-01", periods=n, freq=freq)
    prices = start_price + np.cumsum(np.random.normal(0, 50, n))
    prices = np.abs(prices)

    return pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices * 1.001,
        "volume": np.random.lognormal(5, 1, n),
        "close_time": dates + pd.Timedelta("59min"),
        "quote_asset_volume": np.random.lognormal(10, 1, n),
        "number_of_trades": np.random.randint(10, 1000, n),
        "taker_buy_base_volume": np.random.lognormal(4, 1, n),
        "taker_buy_quote_volume": np.random.lognormal(9, 1, n),
    })


class FakeMarketDataClient:
    """Stub client that returns synthetic data."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_historical_market_data(self, symbol, interval, start, end):
        mask = (self._df["open_time"] >= pd.Timestamp(start)) & (self._df["open_time"] <= pd.Timestamp(end))
        return self._df.loc[mask].copy().reset_index(drop=True)


class TestDataCacheSaveLoad:
    def test_roundtrip(self):
        df = _make_kline_df(50)
        client = FakeMarketDataClient(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(client, cache_dir=Path(tmpdir))
            start = df["open_time"].iloc[0].to_pydatetime()
            end = df["open_time"].iloc[-1].to_pydatetime()

            result = cache.get_klines("BTCUSDT", "1h", start, end)
            assert len(result) == len(df)

            # Load again — should come from cache
            result2 = cache.get_klines("BTCUSDT", "1h", start, end)
            assert len(result2) == len(df)
            pd.testing.assert_frame_equal(
                result.reset_index(drop=True),
                result2.reset_index(drop=True),
            )


class TestDeduplication:
    def test_no_duplicates_after_merge(self):
        df = _make_kline_df(100)
        client = FakeMarketDataClient(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(client, cache_dir=Path(tmpdir))
            start = df["open_time"].iloc[0].to_pydatetime()
            mid = df["open_time"].iloc[50].to_pydatetime()
            end = df["open_time"].iloc[-1].to_pydatetime()

            # First fetch: 0-50
            cache.get_klines("BTCUSDT", "1h", start, mid)
            # Second fetch overlapping: 0-100
            result = cache.get_klines("BTCUSDT", "1h", start, end)

            assert result["open_time"].is_unique, "Duplicate open_time found"


class TestImportCSV:
    def test_import_roundtrip(self):
        df = _make_kline_df(30)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "raw.csv"
            df.to_csv(csv_path, index=False)

            client = FakeMarketDataClient(df)
            cache = DataCache(client, cache_dir=Path(tmpdir) / "cache")

            result = cache.import_existing_csv(csv_path, "BTCUSDT", "1m")
            assert len(result) == 30
            assert (Path(tmpdir) / "cache" / "BTCUSDT" / "1m.csv").exists()
