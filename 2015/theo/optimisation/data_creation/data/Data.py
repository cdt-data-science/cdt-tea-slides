__author__ = 'theopavlakou'

class Data(object):
    """
    A class to represent data that will be passed to an optimisation
    algorithm or for predictors.
    """

    def __init__(self, X, Y):
        """
        Initialises the Data with the input data, X, and
        the targets, Y.

        :param X:   an (n, d) matrix of input data.
                    Each row is a d-dimensional data
                    point.
        :param Y:   an (n, m) matrix of targets.
                    Each row is an m-dimensional target.
        """
        self.X = X
        self.Y = Y