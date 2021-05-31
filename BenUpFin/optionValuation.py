import numpy as np
import pandas as pd
from scipy.stats import stats


class Options:

    def __init__(self, s: pd.DataFrame, style: str, typeOption: str, k: float,
                 premium: float, T: float, rf: float = 0, q: float = 0):
        """
        This class aims at computing several information about options (european) such as payoff or various sensitivities.
        @param s: history of the stock price. The input should be a dataframe.
        @param style: sell ("short") or buy ("long")
        @param typeOption: call ("c") or put ("p").
        @param k: Strike price of your option.
        @param premium: Cost of the option you bought or sold.
        @param T: Time until the end of life of the option.
        @param rf: Risk free rate to apply for the calculations.default=0.
        @param q: Dividend yield (if existing). Default=0.
        """
        self.s = s
        self.style = style
        self.typeOption = typeOption
        self.k = k
        self.premium = premium
        self.T = T
        self.rf = rf
        self.q = q

    def payoff(self):
        df = pd.DataFrame(index=self.s.index)

        if self.style == "long" and self.typeOption == "c":
            df["Payoff"] = max(self.s[:, 0] - self.k, 0) - self.premium

        elif self.style == "long" and self.typeOption == "p":
            df["Payoff"] = max(self.k - self.s[:, 0], 0) - self.premium

        elif self.style == "short" and self.typeOption == "c":
            df["Payoff"] = min(self.k - self.s[:, 0], 0) + self.premium

        elif self.style == "short" and self.typeOption == "p":
            df["Payoff"] = min(df[:, 0] - self.k, 0) + self.premium

        else:
            raise Exception("Combination (style, type) not existing in this world. Check your input.")

        return df

    def bsm_price(self, sigma):

        # calculate the bsm price of European call and put options
        sigma = float(sigma)
        d1 = (np.log(self.s / self.k) + (self.rf - self.q + sigma ** 2 * 0.5) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.typeOption == 'c':
            price = np.exp(-self.rf * self.T) * (self.s * np.exp((self.rf - self.q) * self.T) * stats.norm.cdf(d1)
                                                 - self.k * stats.norm.cdf(d2))
            return price

        elif self.typeOption == 'p':
            price = np.exp(-self.rf * self.T) * (self.k * stats.norm.cdf(-d2) -
                                                 self.s * np.exp((self.rf - self.q) * self.T) * stats.norm.cdf(-d1))
            return price

        else:
            print(f"No such option type, check function parameters")

    def delta(self):
        return

    def gamma(self):
        return

    def vega(self):
        return

    def theta(self):
        return

    def rho(self):
        return



    def compute_implied_volatility(self):
        # apply bisection method to get the implied volatility by solving the BSM function
        precision = 0.00001
        upper_vol = 500.0
        max_vol = 500.0
        lower_vol = 0.0001
        iteration = 0

        while 1:
            iteration += 1
            mid_vol = (upper_vol + lower_vol) / 2.0
            price = self.bsm_price(self, mid_vol)

            if self.typeOption == 'c':

                lower_price = self.bsm_price(self, lower_vol)
                if (lower_price - self.premium) * (price - self.premium) > 0:
                    lower_vol = mid_vol
                else:
                    upper_vol = mid_vol
                if abs(price - self.premium) < precision:
                    break
                if iteration > 50:
                    break

            elif self.typeOption == 'p':
                upper_price = self.bsm_price(self, upper_vol)

                if (upper_price - self.premium) * (price - self.premium) > 0:
                    upper_vol = mid_vol
                else:
                    lower_vol = mid_vol
                if abs(price - self.premium) < precision:
                    break


        return mid_vol
