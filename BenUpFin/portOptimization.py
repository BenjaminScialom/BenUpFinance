import pandas as pd
import numpy as np
from numpy.random import dirichlet, choice
import matplotlib.pyplot as plt
from BenUpFin import preProcessing as pp


def plot_sim_portfolios(simul_perf: pd.DataFrame(), nb_simulation: int = 10000):
    """
    Plotting function of the randomly generated portfolios.
    @param simul_perf: First out put of the sim_portfolios method.
    @param nb_simulation: nb of simulated portfolios generated.
    @return: A plot
    """
    print(simul_perf)
    ax = simul_perf.plot.scatter(x=0, y=1, c=2, cmap='Blues',
                                 alpha=0.2, figsize=(14, 9), colorbar=True,
                                 title=f'{nb_simulation:,d} Simulated Portfolios')

    max_sharpe_idx = simul_perf.iloc[:, 2].idxmax()
    sd, r = simul_perf.iloc[max_sharpe_idx, :2].values
    print(f'Max Sharpe: annualized standard deviation={sd:.2%}, anualized return={r:.2%}')
    ax.scatter(sd, r, marker='*', color='darkblue', s=500, label='Max. Sharpe Ratio')

    min_vol_idx = simul_perf.iloc[:, 0].idxmin()
    sd, r = simul_perf.iloc[min_vol_idx, :2].values
    ax.scatter(sd, r, marker='*', color='green', s=500, label='Min Volatility')
    plt.legend(labelspacing=1, loc='upper left')
    plt.tight_layout()
    plt.show()


class PortOpt:

    def __init__(self, data: pd.DataFrame(), tickers: [str]):
        self.tickers = tickers
        self.data = data
        self.returns = pp.get_daily_returns(data=data, tickers=tickers, method="percent")

    def get_days_past(self) -> float:
        """
        Calculate the numbers of years past according to the index of the dataframe.
        It is use for annualization of the volatility
        """
        start_date = self.returns.index[0]
        end_date = self.returns.index[-1]

        return round((end_date - start_date).days, 4)

    def sim_portfolios(self, nb_simulation: int = 10000, rf_rate: float = 0, short=False):
        """
        The simulation generates random weights using the Dirichlet distribution, and computes the mean,
        standard deviation, and SR for each sample portfolio using the historical return data.
        @param rf_rate: risk free rate which typically is the 3-month treasury bond rate. default=0
        @param nb_simulation: Number of simulation to run. Number of random portfolio to generate.
        @param short: if some position are shorted set it to true (idest: negative weights possible)
        @return:Dataframe and vectors of weights.
        """

        returns = []
        volatilities = []
        sharpes = []
        alpha = np.full(shape=len(self.returns.columns), fill_value=.05)

        for i in range(nb_simulation):

            # Generate weights (randomly)
            weights = dirichlet(alpha=alpha)
            if short:
                weights *= choice([-1, 1])
            # Generate associated returns and volatilities
            rp = np.sum(self.returns.mean() * weights)*self.get_days_past()
            var_p = np.dot(weights.T, np.dot(self.returns.cov(), weights))
            std_p = np.sqrt(var_p) * np.sqrt(self.get_days_past())
            sharpe_p = (rp - rf_rate) / std_p
            returns.append(rp)
            sharpes.append(sharpe_p)
            volatilities.append(std_p)

        # Convert lists to arrays
        returns = np.array(returns)
        volatilities = np.array(volatilities)
        sharpes = np.array(sharpes)

        return pd.DataFrame({'Annualized Standard Deviation': volatilities,
                             'Annualized Returns': returns,
                             'Sharpe Ratio': sharpes})



