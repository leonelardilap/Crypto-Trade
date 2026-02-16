import enum
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


class RegimeType(enum.Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    regime: RegimeType
    confidence: float          # 0-1
    timestamp: datetime | None
    volatility: float
    trend_strength: float      # positive = up, negative = down
    mean_reversion_score: float


class RegimeDetector:
    """Classify market regime using volatility, trend, variance ratio, and Hurst exponent."""

    def __init__(
        self,
        vol_window: int = 20,
        trend_window: int = 50,
        vr_window: int = 20,
        hurst_window: int = 100,
        vol_high_threshold: float = 0.03,
        trend_threshold: float = 0.02,
    ):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vr_window = vr_window
        self.hurst_window = hurst_window
        self.vol_high_threshold = vol_high_threshold
        self.trend_threshold = trend_threshold

    # ---- internal computations ----

    @staticmethod
    def compute_rolling_volatility(prices: pd.Series, window: int) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.rolling(window).std()

    @staticmethod
    def compute_trend_strength(prices: pd.Series, window: int) -> pd.Series:
        """(price - SMA) / SMA — positive means above trend."""
        sma = prices.rolling(window).mean()
        return (prices - sma) / sma

    @staticmethod
    def compute_variance_ratio(prices: pd.Series, window: int) -> float:
        """Variance ratio test: VR(q) = Var(q-period returns) / (q * Var(1-period return)).

        VR ≈ 1 → random walk, VR < 1 → mean-reverting, VR > 1 → trending.
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) < window * 2:
            return 1.0
        var_1 = log_returns.var()
        if var_1 == 0:
            return 1.0
        q_returns = np.log(prices / prices.shift(window)).dropna()
        var_q = q_returns.var()
        return var_q / (window * var_1)

    @staticmethod
    def compute_hurst_exponent(prices: pd.Series, max_lag: int | None = None) -> float:
        """Simplified R/S Hurst exponent. H < 0.5 → mean-reverting, H > 0.5 → trending."""
        ts = np.log(prices / prices.shift(1)).dropna().values
        n = len(ts)
        if n < 20:
            return 0.5

        if max_lag is None:
            max_lag = min(n // 2, 100)

        lags = range(2, max_lag + 1)
        rs_values = []
        for lag in lags:
            subseries = ts[:lag]
            mean_sub = subseries.mean()
            deviate = np.cumsum(subseries - mean_sub)
            r = deviate.max() - deviate.min()
            s = subseries.std(ddof=1)
            if s > 0:
                rs_values.append(r / s)
            else:
                rs_values.append(0)

        valid = [(l, rs) for l, rs in zip(lags, rs_values) if rs > 0]
        if len(valid) < 2:
            return 0.5

        log_lags = np.log([v[0] for v in valid])
        log_rs = np.log([v[1] for v in valid])
        poly = np.polyfit(log_lags, log_rs, 1)
        return float(poly[0])

    # ---- main API ----

    def detect(self, df: pd.DataFrame, price_col: str = "close", time_col: str = "open_time") -> RegimeState:
        """Classify the current (latest) regime from a price DataFrame."""
        prices = df[price_col].astype(float)
        timestamp = df[time_col].iloc[-1] if time_col in df.columns else None

        vol = self.compute_rolling_volatility(prices, self.vol_window)
        current_vol = vol.iloc[-1] if not vol.empty else 0.0

        trend = self.compute_trend_strength(prices, self.trend_window)
        current_trend = trend.iloc[-1] if not trend.empty else 0.0

        tail = prices.tail(max(self.vr_window * 2, self.hurst_window))
        vr = self.compute_variance_ratio(tail, self.vr_window)
        hurst = self.compute_hurst_exponent(tail)

        mr_score = (1.0 - vr) * 0.5 + (0.5 - hurst)  # higher → more mean-reverting

        regime, confidence = self._classify(current_vol, current_trend, vr, hurst)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            timestamp=timestamp,
            volatility=float(current_vol) if np.isfinite(current_vol) else 0.0,
            trend_strength=float(current_trend) if np.isfinite(current_trend) else 0.0,
            mean_reversion_score=float(mr_score) if np.isfinite(mr_score) else 0.0,
        )

    def detect_regime_series(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        time_col: str = "open_time",
        step: int = 1,
    ) -> pd.DataFrame:
        """Compute regime at each row (or every `step` rows) for backtesting."""
        min_rows = max(self.vol_window, self.trend_window, self.vr_window * 2, self.hurst_window) + 1
        records = []

        for i in range(min_rows, len(df), step):
            window = df.iloc[:i + 1]
            state = self.detect(window, price_col, time_col)
            records.append({
                "index": i,
                time_col: df[time_col].iloc[i] if time_col in df.columns else i,
                "regime": state.regime.value,
                "confidence": state.confidence,
                "volatility": state.volatility,
                "trend_strength": state.trend_strength,
                "mean_reversion_score": state.mean_reversion_score,
            })

        return pd.DataFrame(records)

    def _classify(self, vol: float, trend: float, vr: float, hurst: float):
        """Return (RegimeType, confidence) based on indicators."""
        if not np.isfinite(vol):
            return RegimeType.UNKNOWN, 0.1

        # High volatility dominates
        if vol > self.vol_high_threshold:
            confidence = min(1.0, vol / self.vol_high_threshold * 0.5)
            return RegimeType.HIGH_VOLATILITY, confidence

        # Mean-reverting: VR < 0.8 and Hurst < 0.45
        if vr < 0.8 and hurst < 0.45:
            confidence = min(1.0, (0.8 - vr) + (0.45 - hurst))
            return RegimeType.MEAN_REVERTING, max(0.1, confidence)

        # Trending
        if abs(trend) > self.trend_threshold:
            confidence = min(1.0, abs(trend) / self.trend_threshold * 0.5)
            if trend > 0:
                return RegimeType.TRENDING_UP, confidence
            else:
                return RegimeType.TRENDING_DOWN, confidence

        # Weak trending signals from Hurst
        if hurst > 0.55 and abs(trend) > self.trend_threshold * 0.5:
            if trend > 0:
                return RegimeType.TRENDING_UP, 0.3
            else:
                return RegimeType.TRENDING_DOWN, 0.3

        return RegimeType.UNKNOWN, 0.1
