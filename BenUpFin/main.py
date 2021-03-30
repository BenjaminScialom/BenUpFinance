from BenUpFin import portOptimization
from BenUpFin import perfMetrics
from BenUpFin import preProcessing
from BenUpFin import riskMetrics
import yfinance as yf
import pandas as pd


def main():
    tickers = ["AAPL", "MSFT"]
    data = yf.download(tickers=tickers, period="1y")
    risk = riskMetrics.Metrics(data, tickers, [0.5, 0.5])
    es_p = risk.historicalPortfolioES()
    var_p = risk.historicalPortfolioVaR()
    var = risk.historicalVaR()
    es = risk.historicalExpectedShortfall()
    param_var_normal = risk.parametricVar_Normal()
    param_es_normal = risk.parametricES_Normal()
    param_var_student = risk.parametricVar_student(5)
    param_es_student = risk.parametricES_student(5)
    var_mc_p = risk.MonteCarlo_Portfolio_VaR()
    es_mc_p = risk.MonteCarlo_Portfolio_ES()
    es_mc = risk.MonteCarlo_ES()
    var_mc = risk.MonteCarlo_VaR()
    print("Detailed view:")
    print(f"VaR = {var}")
    print(f"Expected Shortfall = {es}")
    print("-"*50)
    print("Portfolio view:")
    print(f"Portfolio VaR = {var_p}")
    print(f"Portfolio Expected Shortfall = {es_p}")
    print("-"*50)
    print("Parametric (normal) view:")
    print(f"VaR = {param_var_normal}")
    print(f"Expected Shortfall = {param_es_normal}")
    print("-"*50)
    print("Parametric (student) view:")
    print(f"VaR = {param_var_student}")
    print(f"Expected Shortfall = {param_es_student}")
    print("-"*50)
    print("Portfolio Monte-Carlo view:")
    print(f"Portfolio VaR = {var_mc_p}")
    print(f"Portfolio Expected Shortfall = {es_mc_p}")
    print("-"*50)
    print("Detailed Monte-Carlo view:")
    print(f"VaR = {var_mc}")
    print(f"Expected Shortfall = {es_mc}")



if __name__ == '__main__':
    main()
