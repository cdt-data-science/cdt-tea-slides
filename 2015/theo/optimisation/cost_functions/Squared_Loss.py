__author__ = 's1463104'

from Cost_Function import Cost_Function
from auxiliary.common_functions.Mean_Squared_Error import Mean_Squared_Error

import numpy as np
rng = np.random

class Squared_Loss(Cost_Function):
    """
    A concrete class (i.e. a class from which an object can be created).
    This class implements the squared loss cost function i.e.

            f_i(w) = 1/2*(dot(w, x_i) - y_i)
    """

    def loss(self, w, indices=[]):
        """
        This overrides the abstract class implementation (which does
        nothing). It makes it specific to the squared loss cost
        function.

        :param w:   A (self.dimension, ) vector. These are the
                    parameters.
        :param indices: A list of indices for the data points
                        which are to be considered for the loss.
                        If the list is empty, all data points
                        are considered.
        :return:    The loss which is a scalar.
        """
        if len(indices) == 0:
            y_hat = np.dot(self.X, w)
            data_dependent_cost = Mean_Squared_Error.evaluate(y_hat, self.y)
        else:
            y_hat = np.dot(self.X[indices, :], w)
            data_dependent_cost = Mean_Squared_Error.evaluate(y_hat, self.y[indices])
        return self.l/2*np.sum(w**2) + data_dependent_cost

    def derivative(self, w, indices=[]):
        """
        This overrides the abstract class implementation (which does
        nothing). It makes it specific to the squared loss cost
        function.

        :param w:   A (self.dimension, ) vector. These are the
                    parameters.
        :param indices: A list of indices for the data points
                        which are to be considered for the
                        gradient of the loss function.
                        If the list is empty, all data points
                        are considered.
        :return:    The gradient of the loss with respect
                    to w for the functions specified by
                    indices. This is a (self.dimension, )
                    vector.
        """
        if len(indices) == 0:
            data_dependent_gradient = np.dot(self.X.T, Mean_Squared_Error.gradient(np.dot(self.X, w), self.y))
        else:
            data_dependent_gradient = np.dot(self.X[indices, :].T, Mean_Squared_Error.gradient(np.dot(self.X[indices, :], w), self.y[indices]))
        return self.l*w + data_dependent_gradient
