__author__ = 'theopavlakou'
import numpy as np

class Mean_Squared_Error(object):
    """
    A class that represents the mean squared error function i.e.

            g(x, y) = 0.5*mean(x - y)**2
    """

    @staticmethod
    def evaluate(x, y):
        """
        Evaluates the mean squared error of two inputs.
        NOTE: It is static because there is no state that is necessary.

        :param x:   An (m, ) vector representing the approximation.
        :param y:   An (m, ) vector which is the target.
        :return:    A scalar which represents the squared error.
        """
        return 0.5*np.mean((x - y)**2)

    @staticmethod
    def gradient(x, y):
        """
        Evaluates the gradient of the mean squared error of two inputs.
        NOTE: It is static because there is no state that is necessary.

        :param x:   An (m, ) vector representing the approximation.
        :param y:   An (m, ) vector which is the target.
        :return:    An (m, ) vector which is the gradient with respect
                    to x.
        """
        return 1/len(x)*(x-y)