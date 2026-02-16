import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Existing env vars (preserved)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL")

# Testnet / safety
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Risk parameters
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
