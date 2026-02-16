# Crypto-Trade

Adaptive quantitative trading system for cryptocurrency markets. Combines regime detection, Kelly criterion position sizing, and event-driven backtesting to optimize BTC/USDT trading on Binance.

## Features

- **Regime Detection** — Classifies market state (trending, mean-reverting, high-volatility) using rolling volatility, variance ratio test, and Hurst exponent
- **Kelly Criterion Sizing** — Optimal position sizing with regime-adjusted scaling, drawdown circuit breaker, and kill switch
- **Backtesting Engine** — Event-driven simulation with realistic fees, slippage, and balance validation
- **Strategies** — Momentum (SMA crossover + volume confirmation) and mean-reversion (z-score based) with regime-aware activation
- **Distribution Fitting** — Fits 14+ scipy distributions to returns with KS test, AIC, and BIC model selection
- **Risk Management** — Max risk cap (2%), max position cap (10%), drawdown breaker (15%), min notional enforcement
- **Data Infrastructure** — Local kline cache with incremental updates, rate limiter, exponential backoff retry

## Setup

```bash
# Clone and install
git clone https://github.com/leonelardilap/Crypto-Trade.git
cd Crypto-Trade
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Binance API keys
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | — | Binance API key |
| `BINANCE_SECRET_KEY` | — | Binance API secret |
| `TRADE_SYMBOL` | `BTCUSDT` | Trading pair |
| `BINANCE_TESTNET` | `true` | Use testnet (safe by default) |
| `DRY_RUN` | `true` | Simulate without executing trades |
| `MAX_RISK_PER_TRADE` | `0.02` | Max portfolio fraction per trade |
| `KELLY_FRACTION` | `0.25` | Kelly fraction scaling |
| `MAX_POSITION_PCT` | `0.10` | Max total position as % of portfolio |

## Usage

### CLI

```bash
# Run a backtest with momentum strategy
python -m src.main backtest --strategy momentum --capital 10000 --file data/raw/market/BTCUSDT_1h_data.csv

# Run with mean-reversion strategy
python -m src.main backtest --strategy mean_reversion --start 2023-01-01 --end 2024-01-01

# Monitor current market regime
python -m src.main monitor --symbol BTCUSDT --interval 1h

# Paper trading (dry-run enforced)
python -m src.main paper-trade --symbol BTCUSDT
```

### Python API

```python
from src.analysis.regime import RegimeDetector
from src.evaluation.backtest import BacktestEngine
from src.strategies.momentum import MomentumStrategy
from src.trade_executor.risk_manager import RiskManager
from src.fetch.market_data import MarketDataClient

# Load data
client = MarketDataClient()
data = client.read_historical_market_data("data/raw/market/BTCUSDT_1h_data.csv")

# Run backtest
engine = BacktestEngine(initial_capital=10_000)
strategy = MomentumStrategy()
result = engine.run(data, strategy.generate_signal, warmup_periods=100)

print(f"Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe: {result.metrics['sharpe']:.2f}")
print(f"Max DD: {result.metrics['max_drawdown']:.2%}")
```

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/
  config.py                  # Environment variables, risk params, paths
  main.py                    # CLI entry point (backtest/monitor/paper-trade)
  analysis/
    regime.py                # RegimeDetector (volatility, trend, Hurst, VR)
    returns.py               # DistributionFitter (14+ distributions, KS/AIC/BIC)
    price.py                 # Power-law regression analysis
    kde.py                   # Kernel density estimation
  evaluation/
    backtest.py              # BacktestEngine (event-driven simulation)
    metrics.py               # MarketMetrics (Sharpe, Sortino, Calmar, drawdown)
    logger.py                # Structured logging + TradeJournal (JSONL)
  fetch/
    market_data.py           # MarketDataClient (Binance API, rate limiting)
    data_cache.py            # DataCache (local CSV cache, incremental updates)
  strategies/
    base.py                  # BaseStrategy ABC
    momentum.py              # MomentumStrategy (SMA crossover + volume)
    mean_reversion.py        # MeanReversionStrategy (z-score)
    DCA.py                   # Dollar-cost averaging backtest
  trade_executor/
    binance_client.py        # Client factory (testnet support, RateLimiter)
    risk_manager.py          # RiskManager (Kelly, drawdown breaker, kill switch)
    trade_logic.py           # Order execution (validation, dry-run)
    portfolio_manager.py     # Balance retrieval, trends, volatility
    market_ops.py            # VWAP quotation from order book
  utils/
    constants.py             # Binance constants (fees, precision, rate limits)
    timers.py                # @timed decorator
tests/
  conftest.py                # Shared fixtures (sample OHLCV, returns, regime)
  test_risk_manager.py       # Kelly, position sizing, drawdown, kill switch
  test_regime.py             # Trending/mean-reverting/high-vol detection
  test_backtest.py           # Buy-and-hold, fees, balance constraints
  test_metrics.py            # Sharpe, Sortino, Calmar, win rate
  test_data_cache.py         # Cache roundtrip, dedup, CSV import
```

## Mathematical Framework

### Market Model

The value of BTC is modeled as a discrete time series $`\{X_t\}`$ of its price in USDT. A trade buying at time $`t_i`$ and selling at $`t_j`$ yields return:

```math
\phi_i = \frac{X_{t_j}}{X_{t_i}} - 1
```

### Cumulative Returns

Investing fraction $`f_i`$ of portfolio $`\beta`$ on each trade:

```math
\beta_m = \beta_0 \prod_{i=1}^{m} (1 + f_i \phi_i)
```

### Growth Metric

```math
G = E\left[\ln(1 + f\phi)\right]
```

### Optimal Position Size (Kelly Criterion)

Maximizing $`G`$ via Taylor expansion to second order:

```math
f^* \approx \frac{E[\phi]}{E[\phi^2]}
```

This is implemented in `RiskManager.compute_kelly_fraction()`, then adjusted by regime confidence and capped by risk limits.

## License

MIT
