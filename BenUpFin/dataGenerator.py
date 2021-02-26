import pandas as pd
import numpy as np
import yfinance as yf


class Data:

    def __init__(self, tickers: [str]):
        self.tickers = tickers

    def oclhv(self, period: str) -> pd.DataFrame():
        """
        Get historical data for a given period until the current day.
        @param period: string (ex: 1y for 1 year until this day)
        @return: 2D Dataframe
        """
        return yf.download(self.tickers, period=period).dropna()

    def oclhv_start_end(self, start: str, end: str) -> pd.DataFrame():
        """
        Get historical data for all the tickers between 2 dates
        @param start: "yyy-mm-dd" or "yyyy"
        @param end: "yyy-mm-dd" or "yyyy"
        @return: 2D Dataframe
        """
        return yf.download(self.tickers, start=start, end=end).dropna()

    def get_close_only(self, df: pd.DataFrame()) -> pd.DataFrame:
        """
        Get the close (adjusted) for every tickers
        @param df: A 2D dataframe must be passed with one of the column named 'Adj Close'.
        @return: 1D Dataframe where the column names are the tickers and the data are the adjusted close.
        """

        adj_close = pd.DataFrame()
        for name in self.tickers:
            adj_close[name] = df['Adj Close'][name]
        return adj_close.dropna()

    def get_close_returns(self, df: pd.DataFrame(), period: int = 1, method: str = "percent"):
        """

        @param df: Dataframe (1D) of prices with a column name "Adj Close"
        @param period: period over which the returns have to be computed
        @param method: log if you want log returns or percent if you want the basic return (as percentage of change)
        @return: Dataframe of returns
        """
        returns = pd.DataFrame(index=df.index)
        for name in self.tickers:
            if method == "percent":
                returns['Returns'] = df['Adj Close'][name].pct_change(period)
            elif method == "log":
                returns['Log Returns'] = np.log(1 + df['Adj Close'][name].pct_change(period))

        return returns.dropna()

    @staticmethod
    def get_risk_free_rate() -> float:
        """
        Get the 3-month treasury bond rate which is the risk free rate.
        @return: mean 3 month treasury bond rate over 1 year
        """
        rf_rate = yf.download(tickers="^IRX", period="6m")["Adj Close"].dropna().mean()
        return round(rf_rate, 5)
