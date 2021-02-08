from typing import Any, Dict
import numpy as np
import pandas as pd


class PerfMetrics:

    def __init__(self, prices: pd.DataFrame, method: str = 'dollar'):
        self.prices = prices
        self.method = method



    def maxdrawdown(self) -> Dict[str, Any]:

        max_dd = 0
        local_peak_date = peak_date = through_date = self.prices.index[0]
        local_peak_price = peak_price = through_price = self.prices.iloc[0]


        for date, price in self.prices.iteritems():

            # keep track of the rolling max
            if price > local_peak_price:
                local_peak_date = date
                local_peak_price = price

            # Compute draw downs
            if self.method == 'dollar':
                dd = local_peak_price - price
            elif self.method == 'percent':
                dd = -((price / local_peak_price) - 1)
            else:
                dd = np.log(local_peak_price) - np.log(price)

            # Store new  max draw down values
            if dd > max_dd:
                max_dd = dd
                peak_date = local_peak_date
                peak_price = local_peak_price
                through_date = date
                through_price = price

        return dict(max_drawdown=max_dd, peak_date=peak_date, peak_price=peak_price, trough_price=through_price,
                    through_date=through_date)


