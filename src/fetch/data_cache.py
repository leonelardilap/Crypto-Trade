import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.config import CACHE_DIR
from src.fetch.market_data import MarketDataClient

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]

FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]


class DataCache:
    """Local CSV cache for kline data with incremental update support."""

    def __init__(self, client: MarketDataClient, cache_dir: Path | None = None):
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / symbol / f"{interval}.csv"

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
        for col in FLOAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df

    def _save(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)

    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return new
        if new.empty:
            return existing
        combined = pd.concat([existing, new], ignore_index=True)
        combined.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        combined.sort_values("open_time", inplace=True)
        return combined.reset_index(drop=True)

    def get_klines(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return cached data, fetching only missing ranges from API."""
        path = self._cache_path(symbol, interval)
        cached = self._load(path)

        if cached.empty:
            logger.info("No cache for %s/%s, fetching full range", symbol, interval)
            fetched = self.client.get_historical_market_data(symbol, interval, start, end)
            self._save(fetched, path)
            return fetched

        cache_start = cached["open_time"].min()
        cache_end = cached["open_time"].max()

        parts = [cached]

        if start < cache_start:
            logger.info("Fetching pre-cache data: %s to %s", start, cache_start)
            pre = self.client.get_historical_market_data(symbol, interval, start, cache_start)
            parts.append(pre)

        if end > cache_end:
            logger.info("Fetching post-cache data: %s to %s", cache_end, end)
            post = self.client.get_historical_market_data(symbol, interval, cache_end, end)
            parts.append(post)

        result = self._merge(cached, pd.concat([p for p in parts if not p.empty], ignore_index=True))
        self._save(result, path)

        mask = (result["open_time"] >= pd.Timestamp(start)) & (result["open_time"] <= pd.Timestamp(end))
        return result.loc[mask].reset_index(drop=True)

    def update_to_now(self, symbol: str, interval: str) -> pd.DataFrame:
        """Incremental update: fetch from last cached timestamp to now."""
        path = self._cache_path(symbol, interval)
        cached = self._load(path)

        if cached.empty:
            raise ValueError(f"No existing cache for {symbol}/{interval}. Use get_klines() first.")

        last_time = cached["open_time"].max()
        now = datetime.now(timezone.utc)
        logger.info("Updating %s/%s from %s to %s", symbol, interval, last_time, now)

        new_data = self.client.get_historical_market_data(symbol, interval, last_time, now)
        result = self._merge(cached, new_data)
        self._save(result, path)
        return result

    def import_existing_csv(self, csv_path: str | Path, symbol: str, interval: str) -> pd.DataFrame:
        """Import an existing raw CSV file into the cache."""
        raw = pd.read_csv(csv_path, parse_dates=["open_time", "close_time"])
        for col in FLOAT_COLS:
            if col in raw.columns:
                raw[col] = raw[col].astype(float)

        path = self._cache_path(symbol, interval)
        cached = self._load(path)
        result = self._merge(cached, raw)
        self._save(result, path)
        logger.info("Imported %d rows from %s into cache", len(raw), csv_path)
        return result
