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

    print("The portfolio Value is 1k")

    var = risk.historicalVaR(95)
    print(var)

    es = risk.historicalExpectedShortfall(95)
    print(es)


if __name__ == '__main__':
    main()
