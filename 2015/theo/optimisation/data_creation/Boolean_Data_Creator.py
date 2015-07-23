__author__ = 'theopavlakou'

from data.Data import Data
from Data_Creator import Data_Creator
import numpy as np

class Boolean_Data_Creator(Data_Creator):

    def create_data(self, N, d):
        """
        Creates the data. This needs to be overrided by
        any concrete class.

        :param N: The number of data points
        :param d: The dimension of the input
        :return:    an (N, m) numpy array where m is the
                    dimension of the output which is
                    determined by the type of data creator.
        """
        X_neg = np.random.randn(np.floor(N/2.0), d) + 2*np.ones((1, d))
        X_pos = np.random.randn(np.ceil(N/2.0), d) - 2*np.ones((1, d))
        X = np.vstack((X_neg, X_pos))

        Y_neg = np.zeros((np.floor(N/2.0), 1))
        Y_pos = np.ones((np.ceil(N/2.0), 1))
        Y = np.vstack((Y_neg, Y_pos))

        return Data(X, Y)