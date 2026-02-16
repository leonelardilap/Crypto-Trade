import numpy as np
import pandas as pd
from src.fetch.market_data  import MarketDataClient

class MarketMetrics:
    def __init__(self):
        """
        Initialize the Metrics object with a DataFrame that must include
        a datetime column 'open_time' and price columns (like 'close').
        """
        pass

    def compute_returns(self, df_market_data, price_col='open', date_col='open_time', return_type='pct', return_period='9m'):
        """
        Calculates returns over a desired time period by estimating the data's base frequency
        using median of time differences. Adds columns for initial/final times and actual delta.

        Parameters:
            df (pd.DataFrame): Input dataframe with at least [price_col, date_col].
            price_col (str): Name of the price column.
            date_col (str): Name of the datetime column.
            return_type (str): 'log' or 'pct'.
            return_period (int): Desired return period as pandas in minutes, e.g. '1H', '1D'.

        Returns:
            pd.DataFrame: With columns ['return', 'time_initial', 'time_final', 'actual_time_diff'] added.
        """
        # Extract only the columns of interest
        df_price_data = df_market_data[[date_col, price_col]]
        # Sort & ensure datetime
        df_returns = df_price_data.sort_values(by=date_col).copy()
        df_returns[date_col] = pd.to_datetime(df_returns[date_col])

        # Compute time deltas between all rows
        time_deltas = df_returns[date_col].diff().dt.total_seconds()
        base_time_diff = time_deltas.median()  # more robust than iloc[1] - iloc[0]

        if pd.isna(base_time_diff) or base_time_diff <= 0:
            raise ValueError("Unable to determine a valid time step from your data.")

        # Convert desired return period to seconds
        #period_seconds = pd.Timedelta(return_period).total_seconds()
        period_seconds = pd.Timedelta(return_period, "m").total_seconds()

        # Calculate how many periods in the data correspond to desired time delta
        periods = max(int(round(period_seconds / base_time_diff)), 1)

        # Prepare requested columns with new names
        df_returns['initial_date'] = df_returns[date_col].shift(periods=periods)
        df_returns['final_date'] = df_returns[date_col]
        df_returns['diff_date'] = (df_returns['final_date'] - df_returns['initial_date']).dt.total_seconds()

        # Compute returns
        if return_type == 'log':
            df_returns['return'] = np.log(df_returns[price_col]).diff(periods=periods)
        elif return_type == 'pct':
            df_returns['return'] = df_returns[price_col].pct_change(periods=periods)
        else:
            raise ValueError("return_type must be 'log' or 'pct'")

        # Optional: short summary
        print(f"Estimated base time step: {base_time_diff:.2f} seconds")
        print(f"Return period '{return_period}' corresponds to {periods} data steps.")

        df_returns.columns = ['date', 'price', 'initial_date', 'final_date', 'diff_date', 'return']
        return df_returns

    def read_returns(self, file_path):
        parse_dates = ['date', 'initial_date', 'final_date']
        dtype = {
            'price'     : float,
            'diff_date' : float,
            'return'    : float
        }
        df = pd.read_csv(
            file_path,
            dtype = dtype,
            parse_dates = parse_dates
        )
        return df

    def compute_drawdown(self, df_market_data, price_col='open', date_col='open_time'):
        """
        Calculates the drawdown series from the running maximum of the price.

        Parameters:
            df_market_data (pd.DataFrame): Data with price and date columns.
            price_col (str): Name of the price column.
            date_col (str): Name of the datetime column.

        Returns:
            pd.DataFrame: With columns ['date', 'price', 'drawdown'].
        """
        df = df_market_data[[date_col, price_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)

        running_max = df[price_col].cummax()
        df['drawdown'] = (df[price_col] - running_max) / running_max

        return df.rename(columns={date_col: 'date', price_col: 'price'})[['date', 'price', 'drawdown']]

    def compute_max_drawdown(self, df_market_data, price_col='open', date_col='open_time'):
        """
        Computes the maximum drawdown value and when it occurred.

        Parameters:
            df_market_data (pd.DataFrame): Data with price and date columns.
            price_col (str): Name of the price column.
            date_col (str): Name of the datetime column.

        Returns:
            dict: {'max_drawdown': float, 'date': Timestamp, 'price': float}
        """
        df_drawdown = self.compute_drawdown(df_market_data, price_col, date_col)
        min_drawdown = df_drawdown['drawdown'].min()
        row = df_drawdown.loc[df_drawdown['drawdown'].idxmin()]

        return {
            'max_drawdown': min_drawdown,
            'date': row['date'],
            'price': row['price']
        }

    def compute_rolling_mean_std(self, df_market_data, price_col='open', date_col='open_time', window='24H'):
        """
        Calculates rolling mean and rolling standard deviation over the price.

        Parameters:
            df_market_data (pd.DataFrame): Data with price and date columns.
            price_col (str): Name of the price column.
            date_col (str): Name of the datetime column.
            window (str): Time window as pandas offset string, e.g. '24H', '7D'.

        Returns:
            pd.DataFrame: With columns ['date', 'rolling_mean', 'rolling_std'].
        """
        df = df_market_data[[date_col, price_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=date_col, inplace=True)
        df.set_index(date_col, inplace=True)

        # Calculate how many periods correspond to the window
        window_seconds = pd.Timedelta(window).total_seconds()
        base_step = df.index.to_series().diff().dt.total_seconds().median()
        periods = max(int(round(window_seconds / base_step)), 1)

        # Rolling computations
        df['rolling_mean'] = df[price_col].rolling(window=periods).mean()
        df['rolling_std'] = df[price_col].rolling(window=periods).std()

        df.reset_index(inplace=True)
        return df.rename(columns={date_col: 'date'})[['date', 'rolling_mean', 'rolling_std']].dropna()

    # ---- New performance metrics ----

    @staticmethod
    def compute_sharpe_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        std = excess.std(ddof=1)
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(periods_per_year))

    @staticmethod
    def compute_sortino_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf") if excess.mean() > 0 else 0.0
        down_std = np.sqrt(np.mean(downside ** 2))
        if down_std == 0:
            return 0.0
        return float(excess.mean() / down_std * np.sqrt(periods_per_year))

    @staticmethod
    def compute_calmar_ratio(df_market_data, returns=None, price_col='open', date_col='open_time', periods_per_year: int = 252) -> float:
        """Annualised return / max drawdown."""
        if returns is not None:
            returns = np.asarray(returns, dtype=float)
            returns = returns[np.isfinite(returns)]
            ann_return = np.mean(returns) * periods_per_year
        else:
            ann_return = 0.0

        prices = df_market_data[price_col].astype(float)
        running_max = prices.cummax()
        drawdowns = (prices - running_max) / running_max
        max_dd = abs(drawdowns.min())
        if max_dd == 0:
            return 0.0
        return float(ann_return / max_dd)

    @staticmethod
    def compute_win_rate(returns) -> float:
        """Fraction of positive returns."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))

    @staticmethod
    def compute_profit_factor(returns) -> float:
        """Gross profits / gross losses."""
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)


if __name__ == '__main__':
    file_path = "/Users/leonelardila/Developer/Crypto-Trade/data/raw/market/BTCUSDT_1m_data.csv"
    market_data_client, metrics = MarketDataClient(), MarketMetrics()
    df_market_data = market_data_client.read_historical_market_data(file_path)

    time_frames = {
        '27m' : 27*4*7*24*60,
        '30m' : 30*4*7*24*60,
        '33m' : 33*4*7*24*60,
        '3y'  : 36*4*7*24*60,
        '39m' : 39*4*7*24*60,
        '42m' : 42*4*7*24*60,
        '45m' : 45*4*7*24*60,
        '4y'  : 48*4*7*24*60,
        '5y'  : 60*4*7*24*60,
        '6y'  : 72*4*7*24*60,
        '7y'  : 84*4*7*24*60,
    }

    for time_frame in time_frames:
        df_returns = metrics.compute_returns(df_market_data, price_col='open', date_col='open_time', return_type='pct', return_period=time_frames[time_frame])
        df_returns.to_csv(f"/Users/leonelardila/Developer/Crypto-Trade/data/processed/returns/{time_frame}.csv", index=False)
