import pandas as pd
import numpy as np
from BenUpFin import preProcessing
from scipy.stats import norm, t
class Metrics:

    def __init__(self, data: pd.DataFrame(), tickers: [str], weights: [float]):
        self.data = data
        self.returns = preProcessing.get_daily_returns(data=data, tickers=tickers, method='percent')
        self.weights = weights
        self.portfolioReturns = preProcessing.get_daily_returns(data=data, tickers=tickers, method='percent') @ weights

    def historicalVaR(self, confidenceLevel: int = 95) -> {}:
        var = {}
        for name in self.returns.columns:
            cut = -1*np.percentile(self.returns[name], 100 - confidenceLevel)
            var[name] = np.round(cut, 3)
        return var

    def historicalExpectedShortfall(self, confidenceLevel: int = 95) -> {}:
        es = {}
        for name in self.returns.columns:
            cut = 1*np.percentile(self.returns[name], 100 - confidenceLevel)
            es[name] = np.round(-1*self.returns[name][self.returns[name] <= cut].mean(), 3)
        return es

    def historicalPortfolioVaR(self, confidenceLevel: int = 95):
        cut = -1 * np.percentile(self.portfolioReturns, 100 - confidenceLevel)
        var = np.round(cut, 3)
        return var

    def historicalPortfolioES(self, confidenceLevel: int = 95):
        cut = 1 * np.percentile(self.portfolioReturns, 100 - confidenceLevel)
        es = np.round(-1 * self.portfolioReturns[self.portfolioReturns <= cut].mean(), 3)
        return es

    def parametricVar_Normal(self,  condidenceLevel: float = 0.95):
        var = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var[name] = round(mu + std * norm.ppf(condidenceLevel), 3)
        return var

    def parametricES_Normal(self, condidenceLevel: float = 0.95):
        es = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            es[name] = round(mu + std * norm.pdf(norm.ppf(condidenceLevel)) * (1-condidenceLevel)**-1, 3)

        return es

    def parametricVar_student(self,dof: int,  condidenceLevel: float = 0.95):
        var_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var_t[name] = round(mu + std * t.ppf(condidenceLevel, dof)*np.sqrt((dof-2)/dof), 3)
        return var_t

    def parametricES_student(self, dof: int, condidenceLevel: float = 0.95):
        es_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            xanu = t.ppf(1-condidenceLevel, dof)
            es_t[name] = round((1 /(1-condidenceLevel)) * (1 - dof) ** (-1) * (dof - 2 + xanu ** 2) * t.pdf(xanu, dof) * std + mu, 3)
        return es_t