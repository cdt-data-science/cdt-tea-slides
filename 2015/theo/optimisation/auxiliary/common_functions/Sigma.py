__author__ = 'theopavlakou'
import numpy as np

class Sigma(object):
    """
    A class that represents the sigmoid function commonly used in
    logistic regression and neural networks.
    """

    @staticmethod
    def evaluate(X):
        """
        Evaluates the sigmoid function element-wise on the input.
        NOTE: It is static because there is no state that is necessary.

        :param X:   An (m, n) matrix representing the input.
        :return:    An (m, n) matrix with each element of the
                    input having been sigmoid-ed.
        """
        return 1.0/(1 + np.exp(-X))

    @staticmethod
    def gradient(X):
        """
        Evaluates the gradient of the sigmoid function element-wise
        with respect to the input.
        NOTE: It is static because there is no state that is necessary.

        :param X:   An (m, n) matrix representing the input.
        :return:    An (m, n) matrix with each element of the
                    being the gradient evaluated at the corresponding
                    input element.
        """
        return Sigma.evaluate(X)*Sigma.evaluate(-X)