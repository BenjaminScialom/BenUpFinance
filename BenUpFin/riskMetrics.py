import pandas as pd
import numpy as np
from BenUpFin import preProcessing
from scipy.stats import norm, t
from arch import arch_model
from arch.__future__ import reindexing
import math

class Metrics:

    def __init__(self, data: pd.DataFrame(), tickers: [str], weights: [float]):
        self.data = data
        self.returns = preProcessing.get_daily_returns(data=data, tickers=tickers, method='percent')
        self.weights = weights
        self.portfolioReturns= preProcessing.get_daily_returns(data=data, tickers=tickers, method='percent') @ weights

    #region Historical method

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

    #endregion

    #region Parametric method

    def parametricVar_Normal(self,  confidenceLevel: float = 0.95)->{}:
        var = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var[name] = round(mu + std * norm.ppf(confidenceLevel), 3)
        return var

    def parametricES_Normal(self, confidenceLevel: float = 0.95)->{}:
        es = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            es[name] = round(mu + std * norm.pdf(norm.ppf(confidenceLevel)) * (1-confidenceLevel)**-1, 3)

        return es

    def parametric_Normal_Portfolio(self, confidenceLevel: float = 0.95)->{}:
        riskMeasures = {}

        mu = np.mean(self.portfolioReturns.values)
        std = np.std(self.portfolioReturns.values)
        riskMeasures['Portfolio VaR'] = round(mu + std * norm.ppf(confidenceLevel), 3)
        riskMeasures['Portfolio ES'] = round(mu + std * norm.pdf(norm.ppf(confidenceLevel)) * (1-confidenceLevel)**-1, 3)

        return riskMeasures

    def parametricVar_student(self,dof: int,  confidenceLevel: float = 0.95)->{}:
        var_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            var_t[name] = round(mu + std * t.ppf(confidenceLevel, dof)*np.sqrt((dof-2)/dof), 3)
        return var_t

    def parametricES_student(self, dof: int, confidenceLevel: float = 0.95)->{}:
        es_t = {}
        for name in self.returns.columns:
            mu = np.mean(self.returns[name])
            std = np.std(self.returns[name])
            xanu = t.ppf(1-confidenceLevel, dof)
            es_t[name] = round((-1 /(1-confidenceLevel)) * (1 - dof) ** (-1) * (dof - 2 + xanu ** 2) * t.pdf(xanu, dof) * std -mu, 3)
        return es_t

    def parametric_Student_Portfolio(self,dof: int, co) -> {}:
        riskMeasures = {}

        mu = np.mean(self.portfolioReturns[name])
        std = np.std(self.portfolioReturns[name])
        riskMeasures['Portfolio VaR'] = round(mu + std * t.ppf(confidenceLevel, dof)*np.sqrt((dof-2)/dof), 3)
        riskMeasures['Portfolio ES'] = round((-1 / (1 - confidenceLevel)) * (1 - dof) ** (-1) * (dof - 2 + xanu ** 2) * t.pdf(xanu, dof) * std - mu, 3)

        return riskMeasures

    def Parametric_Normal_RiskMeasures_Portfolio(self, confidenceLevel: float=0.95, window: int = 100) -> pd.DataFrame():

        var = pd.Series(index=Returns.index, name='VaR')
        es = pd.Series(index=Returns.index, name='ES')
        for i in range(0, len(Returns) - window):
            if i == 0:
                Data = Returns[-window:]
            else:
                Data = Returns[-(window + i):-i]
            mu = np.mean(Data)
            stdev = np.std(Data)
            var[-i - 1] = round(mu + stdev * norm.ppf(Confidence_Interval),3)
            es[-i-1] = round(mu + std * norm.pdf(norm.ppf(confidenceLevel)) * (1 - confidenceLevel) ** -1, 3)

        return df

    #endregion

    #region Monte-Carlo simulation Method

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

    #endregion

    #region Dynamic Methods where volatility is updated through time

    def GARCH_Normal_RiskMeasures_Portfolio(self, confidenceLevel:int = 0.95)->pd.DataFrame():

        df = pd.DataFrame(index=self.portfolioReturns.index)
        print(df.head())
        # Specify and fit a GARCH model
        model = arch_model(self.portfolioReturns, p = 1, q = 1,mean = 'constant', vol = 'GARCH', dist = 't', rescale=False)
        model_fit = model.fit(disp='off',)

        # Make mean & variance forecast
        forecast = model_fit.forecast(start=self.portfolioReturns.index.values[0])
        mu = forecast.mean[self.portfolioReturns.index.values[0]:]
        df['mean'] = mu
        std = np.sqrt(forecast.variance[self.portfolioReturns.index.values[0]:])
        df['Volatility'] = std
        df['VaR'] = mu.values + std.values * norm.ppf(confidenceLevel)
        df['ES'] = mu.values + std.values * norm.pdf(norm.ppf(confidenceLevel)) * (1 - confidenceLevel) ** -1

        return df

    def EWMA_RiskMeasures_Portfolio(self, confidenceLevel: float =0.95, decayFactor: float = 0.94, window: int =100)->pd.DataFrame():

        mu = np.mean(self.portfolioReturns)
        var = pd.Series(index=self.portfolioReturns.index, name='VaR')
        es = pd.Series(index=self.portfolioReturns.index, name='ES')
        std = pd.Series(index=self.portfolioReturns.index, name='volatility')

        ## Defining exponentially smoothed weights components
        decay = np.empty([window, ])
        Weights = np.empty([window, ])
        decay[0] = 1.0
        decay[1] = decayFactor
        Range = range(window)

        # Building weights
        for i in range(2, window):
            decay[i] = decay[1] ** Range[i]
        for i in range(window):
            Weights[i] = decay[i] / sum(decay)

        # squared the returns
        sqrdReturns = self.portfolioReturns ** 2

        ## finding standard deviation on the sliding window before computing risk measures (VaR en Expected Shortfall)
        for i in range(0, len(self.portfolioReturns) - window):
            if i == 0:
                sqrdData = sqrdReturns[-(window):]
            else:
                sqrdData = sqrdReturns[-(window + i):-i]

            EWMAstdev = math.sqrt(sum(Weights * sqrdData))
            std[-i-1] = EWMAstdev
            var[-i - 1] = mu + EWMAstdev * norm.ppf(confidenceLevel)
            es[-i-1] = mu + EWMAstdev * norm.pdf(norm.ppf(confidenceLevel)) * (1 - confidenceLevel) ** -1

        df = pd.concat([std, var, es], axis=1)

        return df.dropna()

    #endregion
