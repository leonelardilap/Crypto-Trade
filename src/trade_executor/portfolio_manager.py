import time
import numpy as np
import pandas as pd
from binance import Client, exceptions
from datetime import datetime, timedelta

def get_binance_balances_df(client: Client, include_zero=False) -> pd.DataFrame:
    """
    Retrieves account balances from Binance and returns them as a DataFrame.

    Args:
        client (Client): Authenticated Binance client.
        include_zero (bool): If False, filters out zero balances.

    Returns:
        pd.DataFrame: DataFrame with 'asset', 'free', 'locked', 'total'.
    """
    try:
        account_info = client.get_account()
        balances = account_info.get('balances', [])
        df = pd.DataFrame(balances)
        df['free'] = pd.to_numeric(df['free'], errors='coerce')
        df['locked'] = pd.to_numeric(df['locked'], errors='coerce')
        df['total'] = df['free'] + df['locked']
        if not include_zero:
            df = df[df['total'] > 0]
        #return df[['asset', 'free', 'locked', 'total']].sort_values(by='total', ascending=False)
        return df
    except exceptions.BinanceAPIException as e:
        print(f"❌ Binance API Error: {e.message}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return pd.DataFrame()
    
    

def compute_trends(df: pd.DataFrame, now=None):
    """
    Computes percentage change from historical close prices to the latest one
    across multiple time frames.
    """
    if now is None:
        now = df['open_time'].max()
    df = df.set_index('open_time').sort_index()
    latest_price = df.loc[now, 'close'] if now in df.index else df.loc[df.index.asof(now), 'close']
    time_frames = {
        "1h": now - timedelta(hours=1),
        "6h": now - timedelta(hours=6),
        "12h": now - timedelta(hours=12),
        "1d": now - timedelta(days=1),
        "1w": now - timedelta(weeks=1),
        "15d": now - timedelta(days=15),
        "1mo": now - timedelta(days=30),
    }
    trends = {}
    for label, ts in time_frames.items():
        asof_time = df.index.asof(ts)
        if pd.isna(asof_time):
            trends[label] = None
        else:
            historical_price = df.loc[asof_time, 'close']
            change = ((latest_price - historical_price) / historical_price) * 100
            trends[label] = round(change, 2)
    return trends

def compute_volatility(df: pd.DataFrame, now=None):
    """
    Compute volatility (std of log returns) over multiple timeframes.
    """
    if now is None:
        now = df['open_time'].max()
    df = df.set_index('open_time').sort_index()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    time_frames = {
        "1h": now - timedelta(hours=1),
        "6h": now - timedelta(hours=6),
        "12h": now - timedelta(hours=12),
        "1d": now - timedelta(days=1),
        "1w": now - timedelta(weeks=1),
        "15d": now - timedelta(days=15),
        "1mo": now - timedelta(days=30),
    }
    volatilities = {}
    for label, start_time in time_frames.items():
        returns = df.loc[start_time:now]['log_return'].dropna()
        if len(returns) > 1:
            std = returns.std()
            vol_percent = std * 100  # optional: annualize if needed
            volatilities[label] = round(vol_percent, 4)
        else:
            volatilities[label] = None  # Not enough data
    return volatilities

def compute_mean_price(df: pd.DataFrame, now=None):
    """
    Compute mean closing price over multiple timeframes.
    """
    if now is None:
        now = df['open_time'].max()
    df = df.set_index('open_time').sort_index()
    time_frames = {
        "1h": now - timedelta(hours=1),
        "6h": now - timedelta(hours=6),
        "12h": now - timedelta(hours=12),
        "1d": now - timedelta(days=1),
        "1w": now - timedelta(weeks=1),
        "15d": now - timedelta(days=15),
        "1mo": now - timedelta(days=30),
    }
    mean_prices = {}
    for label, start_time in time_frames.items():
        price_series = df.loc[start_time:now]["close"].dropna()
        if not price_series.empty:
            mean_price = price_series.mean()
            mean_prices[label] = round(mean_price, 2)
        else:
            mean_prices[label] = None  # Not enough data
    return mean_prices