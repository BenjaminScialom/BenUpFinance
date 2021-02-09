from BenUpFin import perfmetrics as pf
import yfinance as yf


def main():

    df = yf.Ticker("MSFT").history(period="1y")
    df_prices = df['Close']
    m = pf.PerfMetrics(df_prices)

    # max drawdown
    # dic = m.maxdrawdown()
    # print(dic)

    # Get years pas from dataframe
    # years_past = m.get_years_past()
    # print(years_past)

    cagr = m.compound_annual_growth_rate()
    print(cagr)


if __name__ == '__main__':
    main()
