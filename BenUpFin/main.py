from BenUpFin import dataGenerator


def main():
    tickers = ["AAPL", "MSFT"]
    data = dataGenerator.Data(tickers)

    # region Get the data (function tests)
    historical1 = data.oclhv("2y")
    historical2 = data.oclhv_start_end(start="2018-01-01", end="2021-01-01")
    close = data.get_close_only(historical1)
    close_returns = data.get_close_returns(historical1)
    rf = data.get_risk_free_rate()
    # print(historical1.head())
    # print('*'*50)
    # print(historical2.head())
    # print(close.head())
    # print("*"*100)
    # print(close_returns.head())
    # print(rf)
    # endregion



if __name__ == '__main__':
    main()
