import pandas as pd
import numpy as np
from BenUpFin import preProcessing
class Metrics:

    def __init__(self, data: pd.DataFrame(), tickers: [str], weights: [float]):
        self.data = data
        self.returns = preProcessing.get_daily_returns(data=data, tickers=tickers, method='percent')
        self.weights = weights

    def historicalVaR(self, confidenceLevel) -> {}:
        var = {}
        for name in self.returns.columns:
            cut = -1*np.percentile(self.returns[name], 100 - confidenceLevel)
            var[name] = np.round(cut, 3)
        return var

    def historicalExpectedShortfall(self, confidenceLevel) -> {}:
        es = {}
        for name in self.returns.columns:
            cut = 1*np.percentile(self.returns[name], 100 - confidenceLevel)
            es[name] = np.round(-1*self.returns[name][self.returns[name] <= cut].mean(), 3)

        return es
