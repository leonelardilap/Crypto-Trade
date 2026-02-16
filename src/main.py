import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

from src.evaluation.logger import setup_logging


def cmd_backtest(args):
    """Run a backtest on historical data."""
    logger = setup_logging("backtest")

    from src.analysis.regime import RegimeDetector
    from src.evaluation.backtest import BacktestEngine, Side
    from src.fetch.market_data import MarketDataClient
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.trade_executor.risk_manager import RiskManager

    client_md = MarketDataClient()

    if args.file:
        logger.info("Loading data from %s", args.file)
        data = client_md.read_historical_market_data(args.file)
    else:
        start = datetime.fromisoformat(args.start) if args.start else datetime.now() - timedelta(days=365)
        end = datetime.fromisoformat(args.end) if args.end else datetime.now()
        logger.info("Fetching %s %s from %s to %s", args.symbol, args.interval, start, end)
        data = client_md.get_historical_market_data(args.symbol, args.interval, start, end)

    risk_manager = RiskManager()
    regime_detector = RegimeDetector()

    engine = BacktestEngine(
        initial_capital=args.capital,
        risk_manager=risk_manager,
        regime_detector=regime_detector,
    )

    strategy = MomentumStrategy() if args.strategy == "momentum" else MeanReversionStrategy()

    result = engine.run(
        data,
        strategy_fn=strategy.generate_signal,
        warmup_periods=max(100, args.warmup),
    )

    print(f"\n{'='*50}")
    print(f"Backtest Results ({args.strategy})")
    print(f"{'='*50}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"  trades: {len(result.trades)}")


def cmd_monitor(args):
    """Monitor current market regime."""
    logger = setup_logging("monitor")

    from src.analysis.regime import RegimeDetector
    from src.fetch.market_data import MarketDataClient

    client_md = MarketDataClient()
    detector = RegimeDetector()

    start = datetime.now() - timedelta(days=30)
    data = client_md.get_historical_market_data(args.symbol, args.interval, start, datetime.now())

    regime = detector.detect(data)
    print(f"\nCurrent regime for {args.symbol}:")
    print(f"  Regime: {regime.regime.value}")
    print(f"  Confidence: {regime.confidence:.2f}")
    print(f"  Volatility: {regime.volatility:.6f}")
    print(f"  Trend strength: {regime.trend_strength:.6f}")
    print(f"  Mean reversion score: {regime.mean_reversion_score:.4f}")


def cmd_paper_trade(args):
    """Paper trading mode (simulation only)."""
    logger = setup_logging("paper_trade")
    logger.info("Paper trading mode — DRY_RUN enforced")
    logger.info("Symbol: %s, Interval: %s", args.symbol, args.interval)
    print("Paper trading is not yet fully implemented. Use --dry-run with backtest for now.")


def main():
    parser = argparse.ArgumentParser(description="Crypto-Trade: Adaptive Quantitative Trading System")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Simulate without execution (default: true)")
    parser.add_argument("--testnet", action="store_true", default=False, help="Use Binance testnet")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", default="1h", help="Kline interval")

    sub = parser.add_subparsers(dest="command")

    # backtest
    bt = sub.add_parser("backtest", help="Run backtest on historical data")
    bt.add_argument("--start", help="Start date (ISO format)")
    bt.add_argument("--end", help="End date (ISO format)")
    bt.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USDT")
    bt.add_argument("--strategy", choices=["momentum", "mean_reversion"], default="momentum")
    bt.add_argument("--warmup", type=int, default=100, help="Warmup periods")
    bt.add_argument("--file", help="Load data from CSV instead of API")

    # monitor
    sub.add_parser("monitor", help="Monitor current market regime")

    # paper-trade
    sub.add_parser("paper-trade", help="Paper trading mode")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "paper-trade":
        cmd_paper_trade(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
