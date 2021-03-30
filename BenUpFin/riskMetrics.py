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

    def historicalPortfolioVaR(self, confidenceLevel: int = 95) -> float:
        cut = -1 * np.percentile(self.portfolioReturns, 100 - confidenceLevel)
        var = np.round(cut, 3)
        return var

    def historicalPortfolioES(self, confidenceLevel: int = 95) -> float:
        cut = 1 * np.percentile(self.portfolioReturns, 100 - confidenceLevel)
        es = np.round(-1 * self.portfolioReturns[self.portfolioReturns <= cut].mean(), 3)
        return es

    def parametricVar_Normal(self,  condidenceLevel: float = 0.95)->{}:
        var = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var[name] = round(mu + std * norm.ppf(condidenceLevel), 3)
        return var

    def parametricES_Normal(self, condidenceLevel: float = 0.95)->{}:
        es = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            es[name] = round(mu + std * norm.pdf(norm.ppf(condidenceLevel)) * (1-condidenceLevel)**-1, 3)

        return es

    def parametricVar_student(self,dof: int,  condidenceLevel: float = 0.95)->{}:
        var_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var_t[name] = round(mu + std * t.ppf(condidenceLevel, dof)*np.sqrt((dof-2)/dof), 3)
        return var_t

    def parametricES_student(self, dof: int, condidenceLevel: float = 0.95)->{}:
        es_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            xanu = t.ppf(1-condidenceLevel, dof)
            es_t[name] = round((-1 /(1-condidenceLevel)) * (1 - dof) ** (-1) * (dof - 2 + xanu ** 2) * t.pdf(xanu, dof) * std -mu, 3)
        return es_t

    def MonteCarlo_Portfolio_VaR(self, nb_simulation = 1000)->float:
        mu = np.mean(self.portfolioReturns)
        std = np.std(self.portfolioReturns)
        T = len(self.portfolioReturns.index)
        sim_returns = []
        for i in range(nb_simulation):
            rand_rets = np.random.normal(mu, std, T)
            sim_returns.append(rand_rets)
        var = round(-np.percentile(sim_returns, 5), 4)

        return var

    def MonteCarlo_Portfolio_ES(self, nb_simulation = 1000, confidenceLevel: int = 95)->float:
        mu = np.mean(self.portfolioReturns)
        std = np.std(self.portfolioReturns)
        T = len(self.portfolioReturns.index)
        sim_returns = []
        for i in range(nb_simulation):
            rand_rets = np.random.normal(mu, std, T)
            sim_returns.append(rand_rets)
        var = np.percentile(sim_returns, 100-confidenceLevel)
        sorted_returns = np.sort(np.array(sim_returns).reshape(-1))
        es = round(-1*sorted_returns[sorted_returns<var].mean(), 4)

        return es

    def MonteCarlo_VaR(self, nb_simulation = 1000, confidenceLevel:int = 95)-> {}:
        var = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            T = len(self.returns.index)
            sim_returns = []
            for i in range(nb_simulation):
                rand_rets = np.random.normal(mu, std, T)
                sim_returns.append(rand_rets)
            var[name] = round(-np.percentile(sim_returns, 100-confidenceLevel), 4)

        return var

    def MonteCarlo_ES(self, nb_simulation = 1000, confidenceLevel:int = 95)-> {}:
        es = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            T = len(self.returns.index)
            sim_returns = []
            for i in range(nb_simulation):
                rand_rets = np.random.normal(mu, std, T)
                sim_returns.append(rand_rets)
            var = np.percentile(sim_returns, 100 - confidenceLevel)
            sorted_returns = np.sort(np.array(sim_returns).reshape(-1))
            es[name] = round(-1 * sorted_returns[sorted_returns < var].mean(), 4)

        return es


