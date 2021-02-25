import pandas as pd
import numpy as np
from numpy.random import random, uniform, dirichlet, choice


class PortOpt:

    def __init__(self, returns: pd.DataFrame(), weights: np.array()):
        self.returns = returns
        self.weights = weights

    def get_years_past(self) -> float:
        """
        Calculate the numbers of years past according to the index of the dataframe.
        It is use for annualization of the volatility
        """
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]

        return round((end_date - start_date).days / 365.25, 4)

    def get_days_past(self) -> float:
        """
        Calculate the numbers of years past according to the index of the dataframe.
        It is use for annualization of the volatility
        """
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]

        return round((end_date - start_date).days, 4)

    def simulate_portfolios(self, nb_simulation: int = 10000, rf_rate: float = 0, short=True):
        """
        The simulation generates random weights using the Dirichlet distribution, and computes the mean,
        standard deviation, and SR for each sample portfolio using the historical return data.
        @param rf_rate: risk-free rate.
        @param nb_simulation: Number of simulation to run. Number of random portfolio to generate.
        @param short: risk free rate which typically is the 3-month treasury bond rate.
        @return:Dataframe and vectors of weights.
        """
        alpha = np.full(shape=len(self.returns.columns), fill_value=.05)
        weights = dirichlet(alpha=alpha, size=nb_simulation)
        if short:
            weights *= choice([-1, 1], size=weights.shape)

        returns = weights @ self.returns.mean().values + 1
        returns *= self.get_days_past()

        std = (weights @ self.returns.T).std()
        std *= np.sqrt(self.get_days_past())

        sharpe = (returns - rf_rate) / std

        return pd.DataFrame({'Annualized Standard Deviation': std,
                             'Annualized Returns': returns,
                             'Sharpe Ratio': sharpe}), weights


