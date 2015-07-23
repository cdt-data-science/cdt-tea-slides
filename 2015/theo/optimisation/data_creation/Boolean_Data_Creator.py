__author__ = 'theopavlakou'

from data.Data import Data
from Data_Creator import Data_Creator
import numpy as np

class Boolean_Data_Creator(Data_Creator):
    """
    Create boolean type data where there are only two classes and the
    data comes from two separate Gaussians.
    """

    def create_data(self, N, d):
        """
        Creates boolean type data. The positive data comes
        from a Gaussian centred at [-1, -1, ..., -1] and the
        negative data from a Gaussian at [1, 1, ..., 1].
        The targets are in {0, 1}, with 0 being the
        negative class and 1 being the positive class.

        :param N: The number of data points
        :param d: The dimension of the input
        :return: a Data object with boolean data.
        """
        X_neg = np.random.randn(np.floor(N/2.0), d) + 2*np.ones((1, d))
        X_pos = np.random.randn(np.ceil(N/2.0), d) - 2*np.ones((1, d))
        X = np.vstack((X_neg, X_pos))

        Y_neg = np.zeros((np.floor(N/2.0), 1))
        Y_pos = np.ones((np.ceil(N/2.0), 1))
        Y = np.vstack((Y_neg, Y_pos))

        return Data(X, Y)