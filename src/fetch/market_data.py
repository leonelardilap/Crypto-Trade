import time
import logging
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.timers import timed
from datetime import datetime, timedelta
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, TRADE_SYMBOL

logger = logging.getLogger(__name__)


class MarketDataClient:
    def __init__(self, client: Client = None, rate_limiter=None):
        if client is None:
            if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
                raise ValueError("Missing Binance API credentials in environment.")
            client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
        self.client = client
        self.rate_limiter = rate_limiter

    def _acquire(self, weight: int = 1):
        if self.rate_limiter is not None:
            self.rate_limiter.acquire(weight)

    def _retry(self, fn, *args, max_retries: int = 3, **kwargs):
        """Call fn with exponential backoff on API errors."""
        for attempt in range(max_retries):
            try:
                self._acquire()
                return fn(*args, **kwargs)
            except BinanceAPIException as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Binance API error (attempt %d/%d): %s. Retrying in %ds",
                               attempt + 1, max_retries, e.message, wait)
                time.sleep(wait)

    def _format_time(self, t):
        if isinstance(t, datetime):
            return t.strftime("%d %b, %Y %H:%M:%S")
        return t

    @timed(label="MarketDataClient")
    def get_historical_market_data(self, symbol: str, interval: str, start_time, end_time) -> pd.DataFrame:
        """
        Fetch historical candlestick (kline) data and return it as a pandas DataFrame.
        Includes open, high, low, close, and volume data with quote and taker breakdowns.
        """
        start_str = self._format_time(start_time)
        end_str = self._format_time(end_time)
        klines = self._retry(self.client.get_historical_klines, symbol, interval, start_str, end_str)

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(klines, columns=columns)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        float_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"
        ]
        df[float_cols] = df[float_cols].astype(float)

        df.drop(columns=["ignore"], inplace=True)
        return df

    @timed(label="MarketDataClient")
    def read_historical_market_data(self, file_path) -> pd.DataFrame:
        parse_dates = ['open_time', 'close_time']
        dtype = {
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
            'quote_asset_volume': float,
            'number_of_trades': int,
            'taker_buy_base_volume': float,
            'taker_buy_quote_volume': float
        }
        df = pd.read_csv(
            file_path,
            dtype = dtype,
            parse_dates = parse_dates
        )
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Fetch the most recent price for a symbol."""
        ticker = self._retry(self.client.get_symbol_ticker, symbol=symbol)
        return float(ticker["price"])

    def get_order_book(self, symbol: str, depth: int = 5) -> dict:
        """Retrieve top bids and asks from the order book."""
        book = self._retry(self.client.get_order_book, symbol=symbol, limit=depth)
        return {
            "bids": [(float(p), float(q)) for p, q in book["bids"]],
            "asks": [(float(p), float(q)) for p, q in book["asks"]],
        }

    def get_recent_trades(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """Retrieve recent trades as a DataFrame."""
        trades = self._retry(self.client.get_recent_trades, symbol=symbol, limit=limit)
        return pd.DataFrame(trades)

if __name__ == '__main__':
    market_data_client = MarketDataClient()
    pair = "ADAUSDT"
    df = market_data_client.get_historical_market_data(
        symbol=pair,
        interval=Client.KLINE_INTERVAL_1DAY,
        start_time=datetime(2017, 8, 17),
        end_time=datetime.now()
    )
    df.to_csv(f"/Users/leonelardila/Developer/Crypto-Trade/data/raw/market/{pair}_{Client.KLINE_INTERVAL_1DAY}_data.csv", index=False)
