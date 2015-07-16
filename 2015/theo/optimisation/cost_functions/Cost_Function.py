__author__ = 'theopavlakou'

class Cost_Function(object):
    """
    An abstract class that provides the interface for cost
    functions. In this context, a cost function is not a
    function of the data points or the targets. It is purely
    a function of the parameters passed to it. Therefore,
    a cost function will have the data points stored and
    the targets and will assume that these are constant.

    We assume that a cost function is composed as such:

            mean( f(w) ) + l/2 * norm(w)**2

    where f(w) is a list of functions, one for each data
    point and each takes an input of dimension d and outputs
    a scalar.

    NOTE: This could easily be extended to vector functions.
    """

    def __init__(self, X, y, l):
        """
        Initialises the cost function.

        :param X:   An (N, d) matrix, with N being the number of
                    data points and d being the dimension of each.
                    These are the data points provided.
        :param y:   A (N, ) vector, where N is the number of data
                    points. These are the targets.
        :param l:   The regularisation constant which is a scalar.
        """
        self.X = X
        self.y = y
        self.l = l
        # 1/N where N is the number of data points
        self.inverse_n = 1.0/self.X.shape[0]
        # The number of data points
        self.num = self.X.shape[0]
        # The dimension of each input
        self.dimension = self.X.shape[1]

    def loss(self, w, indices=[]):
        """
        An abstract method. This describes what is needed
        to get the loss, but not how the loss is calculated.

        :param w:   A (self.dimension, ) vector. These are the
                    parameters.
        :param indices: A list of indices for the data points
                        which are to be considered for the loss.
        :return:    The loss which is a scalar.
        """
        pass

    def gradient(self, w, indices=[]):
        """
        An abstract method. This describes what is needed
        to get the gradient of the loss function, but not
        how the gradient of the loss function is calculated.

        :param w:   A (self.dimension, ) vector. These are the
                    parameters.
        :param indices: A list of indices for the data points
                        which are to be considered for the
                        gradient of the loss function.
        :return:    The gradient of the loss with respect
                    to w for the functions specified by
                    indices. This is a (self.dimension, )
                    vector.
        """
        pass

