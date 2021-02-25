import numpy as np


class Weights:

    def __init__(self, nb_asset) -> np.array():
        """
        Weight object that gives access to different method to generate weights for
        financial analysis.
        @param nb_asset: Number of assets in the portfolio
        @return An array of weights
        """
        self.nb_asset = nb_asset

    def equal_weights(self):
        return np.ones(self.nb_asset) / len(self.nb_asset)

    def random_weights(self):
        weights = np.random.random(self.nb_asset)
        weights /= np.sum(weights)

    

