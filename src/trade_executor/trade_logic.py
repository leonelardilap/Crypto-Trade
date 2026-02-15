import logging
import math

from src.config import DRY_RUN
from src.utils.constants import BTCUSDT_INFO

logger = logging.getLogger(__name__)


def _round_quantity(quantity: float, precision: int = BTCUSDT_INFO.qty_precision) -> float:
    factor = 10 ** precision
    return math.floor(quantity * factor) / factor


def _validate_order(symbol: str, quantity: float, price: float) -> str | None:
    """Return rejection reason or None if valid."""
    notional = quantity * price
    if notional < BTCUSDT_INFO.min_notional:
        return f"notional ${notional:.2f} below min ${BTCUSDT_INFO.min_notional}"
    if quantity <= 0:
        return "quantity must be positive"
    return None


def market_buy(client, symbol, quantity, dry_run: bool | None = None):
    quantity = _round_quantity(quantity)
    price_est = float(client.get_symbol_ticker(symbol=symbol)["price"])

    rejection = _validate_order(symbol, quantity, price_est)
    if rejection:
        logger.warning("BUY rejected: %s", rejection)
        return {"status": "rejected", "reason": rejection}

    if dry_run is None:
        dry_run = DRY_RUN

    if dry_run:
        logger.info("[DRY RUN] BUY %s %s @ ~$%.2f", quantity, symbol, price_est)
        return {
            "status": "dry_run",
            "side": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price_estimate": price_est,
        }

    logger.info("Executing BUY %s %s", quantity, symbol)
    order = client.order_market_buy(symbol=symbol, quantity=quantity)
    return order


def market_sell(client, symbol, quantity, dry_run: bool | None = None):
    quantity = _round_quantity(quantity)
    price_est = float(client.get_symbol_ticker(symbol=symbol)["price"])

    rejection = _validate_order(symbol, quantity, price_est)
    if rejection:
        logger.warning("SELL rejected: %s", rejection)
        return {"status": "rejected", "reason": rejection}

    if dry_run is None:
        dry_run = DRY_RUN

    if dry_run:
        logger.info("[DRY RUN] SELL %s %s @ ~$%.2f", quantity, symbol, price_est)
        return {
            "status": "dry_run",
            "side": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price_estimate": price_est,
        }

    logger.info("Executing SELL %s %s", quantity, symbol)
    order = client.order_market_sell(symbol=symbol, quantity=quantity)
    return order
