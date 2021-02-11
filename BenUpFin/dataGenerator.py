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
        return yf.Tickers(self.tickers).history(period=period)

    def oclhv_start_end(self, start: str, end: str) -> pd.DataFrame():
        """
        Get historical data for all the tickers between 2 dates
        @param start: "yyy-mm-dd" or "yyyy"
        @param end: "yyy-mm-dd" or "yyyy"
        @return: 2D Dataframe
        """
        return yf.Tickers(self.tickers).history(start=start, end=end)

    def get_close_only(self, df: pd.DataFrame()) -> pd.DataFrame:
        """
        Get the close (adjusted) for every tickers
        @param df: A 2D dataframe must be passed with one of the column named 'Adj Close'.
        @return: 1D Dataframe where the column names are the tickers and the data are the adjusted close.
        """

        adj_close = pd.DataFrame()
        for name in self.tickers:
            adj_close[name] = df[name]['Adj Close']
        return adj_close

    @staticmethod
    def get_returns(df: pd.DataFrame(), period: int = 1, method: str = "percent"):
        """

        @param df: Dataframe (1D) of prices with a column name "Adj Close"
        @param period: period over which the returns have to be computed
        @param method: log if you want log returns or percent if you want the basic return (as percentage of change)
        @return: Dataframe of returns
        """
        returns = pd.DataFrame(index=df.index)

        if method == "percent":
            returns['Returns'] = df['Adj Close'].pct_change(period)
        elif method == "log":
            returns['Log Returns'] = np.log(1 + df['Adj Close'].pct_change(period))

        return returns.dropna()
