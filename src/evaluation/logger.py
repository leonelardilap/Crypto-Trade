import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOG_DIR, LOG_LEVEL


def setup_logging(name: str = "crypto_trade", level: str | None = None) -> logging.Logger:
    """Configure console + file logging to logs/ directory."""
    log_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class TradeJournal:
    """Append-only JSONL trade journal."""

    def __init__(self, path: Path | str | None = None):
        if path is None:
            log_dir = Path(LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / "trades.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def record_trade(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        notional: float,
        regime: str = "",
        kelly: float = 0.0,
        **extra,
    ):
        record = {
            "type": "trade",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "notional": notional,
            "regime": regime,
            "kelly": kelly,
            **extra,
        }
        self._append(record)

    def record_snapshot(
        self,
        portfolio_value: float,
        btc_position: float,
        usdt_balance: float,
        regime: str = "",
        **extra,
    ):
        record = {
            "type": "snapshot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": portfolio_value,
            "btc_position": btc_position,
            "usdt_balance": usdt_balance,
            "regime": regime,
            **extra,
        }
        self._append(record)
