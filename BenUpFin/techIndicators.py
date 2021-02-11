import pandas as pd

from BenUpFin.dataGenerator import Data


class Indicators:

    def __init__(self, ticker: str, period: str):
        """
        Technical indicators are function of the market activity for a financial asset.
        They attempt to uncover patters in market behavior using market activity data to produce
        trading signals.
        @param ticker: Symbol of the company to analyze ( ex: Apple -> APPL)
        """
        data = Data(ticker)
        self.series = data.get_close_only(data.oclhv(period))

    def sma(self, period: int = 5) -> pd.DataFrame:
        """
        When the MACD goes below the Signal line, it indicates a sell signal. When it goes above the SignalLine, it indicates a buy signal.
        @param period: Rolling period (number  of days over which the average is calculated. Usual value are 5, 20, 50, 252
        @return: Dataframe on which moving average had been applied. Depending on the period length  (n), the n first index will be dropped.
        """
        return self.series['Close'].rolling(period).mean().dropna()

    def macd(self, period1: int = 5, period2: int = 20) -> pd.DataFrame:
        """
        The MACD oscillator is the difference between two moving average of different lengths.
        period1 must me be inferior or equal to period2. Buy signal when MACD cross zero from below.
        Sell signal when crosses O from above.
        @return: Dataframe of the difference between two moving average with different lengths.
        """
        assert period1 < period2
        df = pd.DataFrame()
        df['MACD'] = self.sma(period1) - self.sma(period2)

        return df

    def bollinger_bands(self, period: int = 20) -> pd.DataFrame:
        """
        Compute the bollinger bands on the the simple moving average (given a period).
        If sma crosses lower band from under, then buy signal. If sma crosses upper band from above, then sell signal.
        @param period: The bollinger band is computed typically on the 20 days moving average.
        @return: Dataframe with 4 columns (sma, lower, upper, signal)
        """
        # Generating the 3 components of the bollinger bands
        df = pd.DataFrame()
        df['sma'] = self.sma(period)
        std_dev = df['sma'].std()
        df['lower'] = df['sma'] - 2 * std_dev
        df['upper'] = df['sma'] + 2 * std_dev

        return df

    def rsi(self, period: int = 14) -> pd.DataFrame():
        """
        RSI ranges from 0 to 100 and generally, when RSI is above 70, it may indicate that the stock is overbought and
        when RSI is below 30, it may indicate the stock is oversold.
        @return: Dataframe with the rsa component. This dataframe is shorter as we lose the n = period first indexes.
        """
        # Creating up and down series based on the original price series
        delta = self.series['close'].diff()
        up_days = delta.copy()
        up_days[delta <= 0] = 0.0
        down_days = abs(delta.copy())
        down_days[delta > 0] = 0.0
        # Getting the mean of the up and down series over a given period
        rs_up = up_days.rolling(period).mean()
        rs_down = down_days.rolling(period).mean()
        rsi = 100 - 100 / (1 + rs_up / rs_down)
        # Create dataframe
        df = pd.DataFrame(index=self.series.index)
        df['RSI'] = rsi

        return df.dropna()

    def stochastic_oscillator(self, period: int = 14) -> pd.DataFrame():
        """
        Stochastic Oscillator follows the speed or the momentum of the price. As a rule, momentum changes before
        the price changes. It measures the level of the closing price relative to low-high range over a period of time.
        @param period: Number of days over which the moving average is computed for low and high prices' series
        @return: Dataframe with data relative to the stochastic oscillator. Shorter than the original series.
        """
        highest_high = self.series['High'].rolling(period).max()
        lowest_low = self.series['Low'].rolling(period).min()
        k = 100 * (self.series['Close'] - lowest_low) / highest_high - lowest_low
        df = pd.DataFrame()
        df['Stochastic Oscillator'] = k
        return df.dropna()

    def price_rate_of_change(self, n: int = 1) -> pd.DataFrame():
        """
        It measures the most recent change in price with respect to the price in n days ago.
        @param n: number of days over which the rate of change is computed. default = 1.
        @return: Dataframe with the price rate of change. the n first indexes are dropped.
        """
        numerator = self.series['Close'].diff(n)
        denumerator = self.series['Close'].shift(n)
        proc = numerator / denumerator
        df = pd.DataFrame()
        df['PROC'] = proc

        return df.dropna()

    def on_balance_volume(self) -> pd.DataFrame():
        """
        OBV rises when volume on up days outpaces volume on down days. OBV falls when volume on down days is stronger.
        A rising OBV reflects positive volume pressure that can lead to higher prices.
        Conversely, falling OBV reflects negative volume pressure that can foreshadow lower prices.
        @return: Dataframe
        """
        i = 0
        obv = [0]
        while i < self.series.index[-1]:
            if self.series.loc[i + 1, 'Close'] - self.series.loc[i, 'Close'] > 0:
                obv.append(self.series.loc[i + 1, 'Volume'])
            if self.series.loc[i + 1, 'Close'] - self.series.loc[i, 'Close'] == 0:
                obv.append(0)
            if self.series.loc[i + 1, 'Close'] - self.series.loc[i, 'Close'] < 0:
                obv.append(-self.series.loc[i + 1, 'Volume'])
            i = i + 1

        df = pd.DataFrame(obv, columns=['OBV'], index=self.series.index)

        return df

    def chaikin_money_flow(self, n: int = 20) -> pd.DataFrame():
        """
        The very basic way to determine a buy signal is when the CMF line crosses above the zero-line
        while the price continues to move upwards. On the contrary, a sell signal is identified when
        the Chaikin Money Flow indicator crosses below the zero-line with the price moving in a downward direction.
        @param n: number of days on which the rolling window is performed
        @return: Dataframe where the n first indexes are dropped (nans).
        """
        mfv = self.series['Volume'] * (2 * self.series['High'] - self.series['Low']) / \
              (self.series['High'] - self.series['Low'])
        c_mfv = mfv.rolling(n).sum() / self.series['Volume'].rolling(n).sum()
        df = pd.DataFrame(c_mfv, columns=['CMF'], index = self.series.index)

        return df.dropna()

