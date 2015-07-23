__author__ = 'theopavlakou'

from data.Data import Data
from Data_Creator import Data_Creator
import numpy as np

class Multi_Class_Data_Creator(Data_Creator):
    """
    Create multi-class type data where there are only m classes and the
    data comes from separate Gaussians.
    """
    def create_data(self, N, d):
        """
        Creates multi class type data.

        :param N: The number of data points
        :param d: The dimension of the input
        :return: a Data object with multi-class data.
        """
        # TODO you get the point
        pass