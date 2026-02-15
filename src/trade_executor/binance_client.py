import time
import threading
from binance.client import Client
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET
from src.utils.constants import TESTNET_BASE_URL, REQUEST_WEIGHT_LIMIT


class RateLimiter:
    """Token-bucket rate limiter for Binance API (1200 weight/minute)."""

    def __init__(self, max_weight: int = REQUEST_WEIGHT_LIMIT, window_seconds: float = 60.0):
        self.max_weight = max_weight
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._tokens = float(max_weight)
        self._last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_weight, self._tokens + elapsed * (self.max_weight / self.window_seconds))
        self._last_refill = now

    def acquire(self, weight: int = 1):
        """Block until enough tokens are available, then consume them."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
            time.sleep(0.05)


def get_binance_client(testnet: bool | None = None) -> Client:
    """Create a Binance client, optionally pointing at testnet.

    Args:
        testnet: If None, uses BINANCE_TESTNET from config.
    """
    if testnet is None:
        testnet = BINANCE_TESTNET

    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

    if testnet:
        client.API_URL = TESTNET_BASE_URL + "/api"

    # Validate connection
    client.ping()
    return client
