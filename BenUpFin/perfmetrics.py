from typing import Any, Dict
import numpy as np
import pandas as pd


def annualized_downside_deviation(returns: pd.DataFrame(), benchmark_rate: float = 0) -> float:
    """
    The benchmark rate is assumed annualized ,so, it needs to be adapted
    according to the number of periods per year seen in the data.
    """
    # adjusting the benchmark
    years_past = returns.get_years_past()
    entries_per_year = returns.shape[0] / years_past
    adj_benchmark_rate = ((1 + benchmark_rate) ** (1/entries_per_year)) - 1

    downside_df = pd.DataFrame()
    downside_df['downside_returns'] = adj_benchmark_rate - returns
    downside_sum_of_squares = downside_df['downside_returns'].apply(lambda x: x**2 if x > 0 else 0).sum()

    downside_deviation = np.sqrt(downside_sum_of_squares/(returns.shape[0] - 1))

    return downside_deviation * np.sqrt(entries_per_year)


class PerfMetrics:

    def __init__(self, prices: pd.DataFrame, method: str = 'dollar'):
        self.prices = prices
        self.method = method


    def calculate_returns(self, period: int = 1, method: str = "percent") -> pd.DataFrame:
        """
        Calculate returns given a dataframe of prices.
        You have the choice between percentage returns or log returns.
        """
        returns = pd.DataFrame()

        if method == "percent":
            returns['Returns'] = self.prices.pct_change(period)
        elif method == "log":
            returns['Log Returns'] = np.log(1 + self.prices.pct_change(period))

        return returns

    def get_years_past(self) -> float:
        """
        Calculate the numbers of years past according to the index of the dataframe.
        It is use for annualization of the volatility
        """
        start_date = self.prices.index[0]
        end_date = self.prices.index[-1]

        return round((end_date - start_date).days/365.25, 4)

    def annualized_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculates annualized volatility for a date-indexed pandas data frame
        It works for any interval of time and whether it is prices or returns.
        """
        years_past = self.get_years_past()
        entries_per_year = df.shape[0]/years_past

        return df.std() * np.sqrt(entries_per_year)

    def compound_annual_growth_rate(self) -> float:
        """
        Calculate compounded annual growth rate of a single asset or a portfolio.
        This is equivalent to annualized returns.
        """
        value_factor = self.prices.iloc[-1] / self.prices.iloc[0]
        years_past = self.get_years_past()

        return round((value_factor ** (1/years_past)) - 1, 4)

    def sharpe_ratio(self, benchmark_rate: float = 0) -> float:
        """
        Calculates sharpe ratio given a pandas dataframe. The benchmark is set to 0 as default.
        If using American securities then it may be better to use the 3-month treasury bond rate.
        """
        cagr = self.compound_annual_growth_rate()
        annual_vol = self.annualized_volatility(self.calculate_returns())

        return (cagr - benchmark_rate) / annual_vol

    def sortino_ratio(self, benchmark_rate: float = 0) -> float:
        """
        Sortino ratio is an improvement on the sharpe ratio.
        It penalizes both large upward and large downward price swings.
        std() built in function cannot be used to compute the downside volatility.
        """
        cagr = self.compound_annual_growth_rate()
        downside_vol = annualized_downside_deviation(self.calculate_returns())
        
        return (cagr - benchmark_rate)/downside_vol

    def maxdrawdown(self) -> Dict[str, Any]:
        """
        Maximum decrease in the signal in the date-indexed of the pandas data frame.
        """
        max_dd = 0
        local_peak_date = peak_date = end_date = self.prices.index[0]
        local_peak_price = peak_price = end_price = self.prices.iloc[0]


        for date, price in self.prices.iteritems():
            print(price)
            print(local_peak_price)
            # keep track of the rolling max
            if price > local_peak_price:
                local_peak_date = date
                local_peak_price = price

            # Compute draw downs
            if self.method == 'dollar':
                dd = local_peak_price - price
            elif self.method == 'percent':
                dd = -((price / local_peak_price) - 1)
            else:
                dd = np.log(local_peak_price) - np.log(price)

            # Store new  max draw down values
            if dd > max_dd:
                max_dd = dd
                peak_date = local_peak_date
                peak_price = local_peak_price
                end_date = date
                end_price = price

        return dict(max_drawdown=max_dd,
                    peak_date=peak_date,
                    peak_price=peak_price,
                    end_date=end_date,
                    end_price=end_price)




