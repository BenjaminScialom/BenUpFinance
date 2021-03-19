from BenUpFin import portOptimization
from BenUpFin import perfMetrics
import yfinance as yf
import pandas as pd


def main():
    tickers = ["AAPL", "MSFT"]
    data = yf.download(tickers=tickers, period="1y")
    # print(data)
    # perfm = perfMetrics.Metrics(data, tickers)
    # df = perfm.get_close_returns()
    # print(df.head())
    portopt = portOptimization.PortOpt(data, tickers)
    simul_perf = portopt.sim_portfolios()
    portOptimization.plot_sim_portfolios(simul_perf)



if __name__ == '__main__':
    main()
