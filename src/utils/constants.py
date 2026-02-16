from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    qty_precision: int
    price_precision: int
    min_notional: float
    step_size: float
    tick_size: float


# BTC/USDT defaults
BTCUSDT_INFO = SymbolInfo(
    symbol="BTCUSDT",
    base_asset="BTC",
    quote_asset="USDT",
    qty_precision=5,
    price_precision=2,
    min_notional=10.0,
    step_size=0.00001,
    tick_size=0.01,
)

# Rate limits
REQUEST_WEIGHT_LIMIT = 1200  # per minute
ORDER_LIMIT_PER_SECOND = 10
ORDER_LIMIT_PER_DAY = 200_000

# Fees
TAKER_FEE = 0.001   # 0.1%
MAKER_FEE = 0.001   # 0.1%

# Kline intervals (subset of python-binance constants)
KLINE_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

# Testnet
TESTNET_BASE_URL = "https://testnet.binance.vision"
