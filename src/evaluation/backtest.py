import enum
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.analysis.regime import RegimeDetector, RegimeState, RegimeType
from src.evaluation.metrics import MarketMetrics
from src.trade_executor.risk_manager import RiskManager
from src.utils.constants import BTCUSDT_INFO

logger = logging.getLogger(__name__)


class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    index: int
    timestamp: object
    side: Side
    price: float
    quantity: float
    notional: float
    fee: float
    regime: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.DataFrame
    regime_series: pd.DataFrame
    metrics: dict = field(default_factory=dict)


# Type alias for strategy function
# strategy_fn(window_df, regime, usdt_balance, btc_position) -> Optional[(Side, returns_array)]
StrategyFn = Callable[[pd.DataFrame, RegimeState, float, float], Optional[tuple[Side, np.ndarray]]]


class BacktestEngine:
    """Event-driven backtesting engine with regime detection and risk management."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 1.0,
        risk_manager: RiskManager | None = None,
        regime_detector: RegimeDetector | None = None,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.risk_manager = risk_manager or RiskManager()
        self.regime_detector = regime_detector or RegimeDetector()

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: StrategyFn,
        price_col: str = "close",
        time_col: str = "open_time",
        warmup_periods: int = 100,
        step: int = 1,
    ) -> BacktestResult:
        """Run a backtest over historical data."""
        data = data.copy().reset_index(drop=True)
        n = len(data)

        usdt_balance = self.initial_capital
        btc_position = 0.0
        trades: list[Trade] = []
        equity_records = []
        regime_records = []

        self.risk_manager.reset_equity(self.initial_capital)

        for i in range(warmup_periods, n, step):
            window = data.iloc[:i + 1]
            price = float(data[price_col].iloc[i])
            timestamp = data[time_col].iloc[i] if time_col in data.columns else i

            # Detect regime
            regime = self.regime_detector.detect(window, price_col, time_col)

            # Portfolio value
            portfolio_value = usdt_balance + btc_position * price

            # Record equity
            equity_records.append({
                time_col: timestamp,
                "equity": portfolio_value,
                "usdt": usdt_balance,
                "btc": btc_position,
                "price": price,
            })

            regime_records.append({
                time_col: timestamp,
                "regime": regime.regime.value,
                "confidence": regime.confidence,
            })

            # Get strategy signal
            signal = strategy_fn(window, regime, usdt_balance, btc_position)
            if signal is None:
                continue

            side, returns_arr = signal

            # Size position via risk manager
            existing_usd = btc_position * price
            pos = self.risk_manager.size_position(
                portfolio_value, price, returns_arr, regime, existing_usd
            )

            if not pos.is_valid:
                continue

            # Apply slippage
            slippage_mult = 1.0 + self.slippage_bps / 10_000
            if side == Side.BUY:
                exec_price = price * slippage_mult
            else:
                exec_price = price / slippage_mult

            quantity = pos.quantity_btc
            notional = quantity * exec_price
            fee = notional * self.fee_rate

            # Execute
            if side == Side.BUY:
                cost = notional + fee
                if cost > usdt_balance:
                    # Scale down to what we can afford
                    affordable = usdt_balance / (exec_price * (1 + self.fee_rate))
                    quantity = _floor_qty(affordable, BTCUSDT_INFO.qty_precision)
                    if quantity <= 0:
                        continue
                    notional = quantity * exec_price
                    fee = notional * self.fee_rate
                    cost = notional + fee

                if notional < BTCUSDT_INFO.min_notional:
                    continue

                usdt_balance -= cost
                btc_position += quantity

            elif side == Side.SELL:
                if quantity > btc_position:
                    quantity = _floor_qty(btc_position, BTCUSDT_INFO.qty_precision)
                    if quantity <= 0:
                        continue
                    notional = quantity * exec_price
                    fee = notional * self.fee_rate

                if notional < BTCUSDT_INFO.min_notional:
                    continue

                usdt_balance += notional - fee
                btc_position -= quantity

            trades.append(Trade(
                index=i,
                timestamp=timestamp,
                side=side,
                price=exec_price,
                quantity=quantity,
                notional=notional,
                fee=fee,
                regime=regime.regime.value,
            ))

        # Final equity
        final_price = float(data[price_col].iloc[-1])
        final_equity = usdt_balance + btc_position * final_price

        equity_df = pd.DataFrame(equity_records)
        regime_df = pd.DataFrame(regime_records)

        # Compute metrics
        metrics_calc = MarketMetrics()
        trade_returns = []
        for t in trades:
            if t.side == Side.SELL:
                # Simplified: use price change as return proxy
                trade_returns.append((t.price - final_price) / final_price)
            else:
                trade_returns.append((final_price - t.price) / t.price)

        if not equity_df.empty:
            equity_returns = equity_df["equity"].pct_change().dropna().values
        else:
            equity_returns = np.array([])

        total_fees = sum(t.fee for t in trades)

        max_dd = 0.0
        if not equity_df.empty:
            running_max = equity_df["equity"].cummax()
            dd = (equity_df["equity"] - running_max) / running_max
            max_dd = float(dd.min())

        result_metrics = {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": (final_equity / self.initial_capital - 1),
            "total_trades": len(trades),
            "total_fees": total_fees,
            "max_drawdown": max_dd,
            "sharpe": metrics_calc.compute_sharpe_ratio(equity_returns),
            "sortino": metrics_calc.compute_sortino_ratio(equity_returns),
            "win_rate": metrics_calc.compute_win_rate(equity_returns),
            "profit_factor": metrics_calc.compute_profit_factor(equity_returns),
        }

        return BacktestResult(
            trades=trades,
            equity_curve=equity_df,
            regime_series=regime_df,
            metrics=result_metrics,
        )


def _floor_qty(qty: float, precision: int) -> float:
    factor = 10 ** precision
    return int(qty * factor) / factor
