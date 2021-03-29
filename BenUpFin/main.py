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

    print("Detailed view:")
    print(var)
    print(es)
    print("Portfolio view:")
    print(var_p)
    print(es_p)
    print("Parametric (normal) view:")
    print(param_var_normal)
    print(param_es_normal)
    print("Parametric (student) view:")
    print(param_var_student)
    print(param_es_student)


if __name__ == '__main__':
    main()
