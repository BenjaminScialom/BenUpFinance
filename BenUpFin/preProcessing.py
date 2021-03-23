import pandas as pd
import numpy as np


def get_daily_returns(data: pd.DataFrame(), tickers: [str], method: str = "percent"):
    """
    @param data: Dataframe with a column name "Adj Close"
    @param tickers: period over which the returns have to be computed
    @param method: log if you want log returns or percent if you want the basic return (as percentage of change)
    @return: Dataframe of returns
    """
    returns = pd.DataFrame()

    for name in tickers:
        if method == "percent":
            returns[name] = data['Adj Close'][name].pct_change()
        elif method == "log":
            returns[name] = np.log(1 + data['Adj Close'][name].pct_change())

    return returns.dropna()

